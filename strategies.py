"""
Domain Adaptation Strategies for Bridging Synthetic-to-Real Gap

Strategy 1: Training-time augmentation (handled in datasets.py)
Strategy 2: Test-Time Augmentation (TTA)
Strategy 3: DANN (Domain Adversarial Neural Network)
Strategy 4: AdaIN Style Transfer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchvision.transforms as transforms
import numpy as np
from typing import Optional, Tuple


# =============================================================================
# Strategy 2: Test-Time Augmentation (TTA)
# =============================================================================

class TestTimeAugmentation:
    """
    Run multiple augmented versions of the input and average point clouds.
    
    No retraining needed — this is a free performance boost.
    
    Usage:
        tta = TestTimeAugmentation(num_augments=10)
        avg_points = tta(model, image)
    """
    
    def __init__(self, num_augments: int = 10, image_size: int = 224):
        self.num_augments = num_augments
        self.image_size = image_size
        
        # Define random augmentations for TTA
        # NOTE: RandomHorizontalFlip REMOVED — flipping image without flipping
        # the predicted point cloud causes misalignment
        self.augment = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.RandomAffine(
                degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)
            ),
        ])
        
        # Denormalize → augment → renormalize
        self.denorm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.denorm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    @torch.no_grad()
    def __call__(self, model, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            model: PointCloudReconstructor
            images: (B, 3, 224, 224) — already normalized
        Returns:
            avg_points: (B, num_points, 3)
        """
        device = images.device
        B = images.shape[0]
        
        all_points = []
        
        # Original prediction
        points = model(images)
        all_points.append(points)
        
        # Augmented predictions
        for _ in range(self.num_augments - 1):
            # Denormalize
            denorm_std = self.denorm_std.to(device)
            denorm_mean = self.denorm_mean.to(device)
            aug_imgs = images * denorm_std + denorm_mean
            aug_imgs = aug_imgs.clamp(0, 1)
            
            # Apply augmentation (per image)
            aug_batch = []
            for i in range(B):
                img_pil = transforms.ToPILImage()(aug_imgs[i].cpu())
                img_aug = self.augment(img_pil)
                img_tensor = transforms.ToTensor()(img_aug)
                # Re-normalize
                img_tensor = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )(img_tensor)
                aug_batch.append(img_tensor)
            
            aug_batch = torch.stack(aug_batch).to(device)
            points = model(aug_batch)
            all_points.append(points)
        
        # Average all predictions
        all_points = torch.stack(all_points, dim=0)  # (num_aug, B, N, 3)
        avg_points = all_points.mean(dim=0)           # (B, N, 3)
        
        return avg_points


# =============================================================================
# Strategy 3: DANN — Domain Adversarial Neural Network
# =============================================================================

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer (GRL).
    Forward: identity
    Backward: negate gradients, scaled by lambda
    """
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wrapper for GRL with adjustable lambda."""
    
    def __init__(self):
        super().__init__()
        self.lambda_val = 1.0
    
    def set_lambda(self, lambda_val: float):
        self.lambda_val = lambda_val
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)


class DomainDiscriminator(nn.Module):
    """
    Binary classifier: is this feature from synthetic or real domain?
    
    Placed after the encoder with a Gradient Reversal Layer.
    Forces encoder to learn domain-invariant features.
    
    Input:  (B, 49, 512) — encoder features
    Output: (B, 1) — domain probability (0=synthetic, 1=real)
    """
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        
        self.grl = GradientReversalLayer()
        
        # Pool spatial tokens → single feature
        self.pool = nn.AdaptiveAvgPool1d(1)  # (B, 512, 49) → (B, 512, 1)
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )
    
    def set_lambda(self, lambda_val: float):
        self.grl.set_lambda(lambda_val)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, 49, 512) — encoder output
        Returns:
            domain_pred: (B, 1) — domain logits
        """
        # GRL: reverse gradients during backprop
        features = self.grl(features)
        
        # Pool: (B, 49, 512) → (B, 512)
        pooled = features.permute(0, 2, 1)      # (B, 512, 49)
        pooled = self.pool(pooled).squeeze(-1)   # (B, 512)
        
        # Classify domain
        domain_pred = self.classifier(pooled)    # (B, 1)
        
        return domain_pred


def dann_lambda_schedule(epoch: int, max_epoch: int, gamma: float = 10.0) -> float:
    """
    DANN lambda schedule from Ganin et al.
    Gradually increases from 0 to 1 during training.
    
    λ = 2 / (1 + exp(-γ * p)) - 1, where p = epoch / max_epoch
    """
    p = epoch / max_epoch
    return 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0


class DANNTrainer:
    """
    Helper class for DANN training loop.
    
    Combines reconstruction loss with domain adversarial loss.
    Total loss = L_reconstruction + λ * L_domain
    """
    
    def __init__(
        self,
        model: nn.Module,
        discriminator: DomainDiscriminator,
        lambda_max: float = 1.0,
    ):
        self.model = model
        self.discriminator = discriminator
        self.lambda_max = lambda_max
        self.domain_criterion = nn.BCEWithLogitsLoss()
    
    def compute_domain_loss(
        self,
        synthetic_images: torch.Tensor,
        real_images: torch.Tensor,
        epoch: int,
        max_epoch: int,
    ) -> torch.Tensor:
        """
        Compute domain adversarial loss.
        
        Args:
            synthetic_images: (B_s, 3, 224, 224)
            real_images: (B_r, 3, 224, 224)
            epoch: current epoch
            max_epoch: total epochs
        
        Returns:
            domain_loss: scalar
        """
        # Compute lambda
        lambda_val = self.lambda_max * dann_lambda_schedule(epoch, max_epoch)
        self.discriminator.set_lambda(lambda_val)
        
        # Get encoder features
        with torch.no_grad():
            syn_features = self.model.get_encoder_features(synthetic_images)
        real_features = self.model.get_encoder_features(real_images)
        
        # Actually, we need gradients for both
        syn_features = self.model.get_encoder_features(synthetic_images)
        real_features = self.model.get_encoder_features(real_images)
        
        # Domain predictions
        syn_domain = self.discriminator(syn_features)
        real_domain = self.discriminator(real_features)
        
        # Labels: 0 = synthetic, 1 = real
        syn_labels = torch.zeros_like(syn_domain)
        real_labels = torch.ones_like(real_domain)
        
        # Domain loss
        domain_loss = self.domain_criterion(syn_domain, syn_labels) + \
                      self.domain_criterion(real_domain, real_labels)
        
        return domain_loss * lambda_val


# =============================================================================
# Strategy 4: AdaIN Style Transfer
# =============================================================================

class AdaINStyleTransfer(nn.Module):
    """
    Adaptive Instance Normalization for style transfer.
    
    At test time: transforms real images to look more synthetic.
    Matches feature statistics (mean/variance) of real images to synthetic ones.
    
    Based on: "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization"
    by Huang & Belongie, ICCV 2017.
    """
    
    def __init__(self):
        super().__init__()
        
        # VGG-19 encoder for feature extraction
        import torchvision.models as models
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        # Split VGG into layers for multi-scale features
        self.enc_1 = vgg[:2]   # relu1_1
        self.enc_2 = vgg[2:7]  # relu2_1
        self.enc_3 = vgg[7:12] # relu3_1
        self.enc_4 = vgg[12:21] # relu4_1
        
        # Freeze VGG
        for param in self.parameters():
            param.requires_grad = False
        
        # Simple decoder (learned)
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 3, 3, 1, 1),
        )
    
    def encode(self, x: torch.Tensor):
        """Extract multi-scale VGG features."""
        h1 = self.enc_1(x)
        h2 = self.enc_2(h1)
        h3 = self.enc_3(h2)
        h4 = self.enc_4(h3)
        return h1, h2, h3, h4
    
    @staticmethod
    def adain(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
        """
        Adaptive Instance Normalization.
        
        Matches the mean and variance of content features to style features.
        """
        B, C = content_feat.shape[:2]
        
        # Content statistics
        content_mean = content_feat.view(B, C, -1).mean(dim=2, keepdim=True)
        content_std = content_feat.view(B, C, -1).std(dim=2, keepdim=True) + 1e-8
        
        # Style statistics
        style_mean = style_feat.view(B, C, -1).mean(dim=2, keepdim=True)
        style_std = style_feat.view(B, C, -1).std(dim=2, keepdim=True) + 1e-8
        
        # Normalize content, then apply style statistics
        normalized = (content_feat.view(B, C, -1) - content_mean) / content_std
        stylized = normalized * style_std + style_mean
        
        return stylized.view_as(content_feat)
    
    @torch.no_grad()
    def transfer(
        self,
        real_image: torch.Tensor,
        style_image: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        Transfer style from synthetic image to real image.
        
        Args:
            real_image:  (B, 3, H, W) — real photo (content)
            style_image: (B, 3, H, W) — synthetic render (style target)
            alpha: interpolation factor (0=content, 1=fully styled)
        
        Returns:
            styled: (B, 3, H, W) — real image styled to look synthetic
        """
        # Encode both
        _, _, _, content_feat = self.encode(real_image)
        _, _, _, style_feat = self.encode(style_image)
        
        # AdaIN
        t = self.adain(content_feat, style_feat)
        
        # Interpolate
        t = alpha * t + (1 - alpha) * content_feat
        
        # Decode
        styled = self.decoder(t)
        
        return styled
    
    @torch.no_grad()
    def transfer_to_synthetic_style(
        self,
        real_images: torch.Tensor,
        synthetic_stats: dict,
    ) -> torch.Tensor:
        """
        Simplified: transfer real images toward pre-computed synthetic statistics.
        
        Args:
            real_images: (B, 3, H, W)
            synthetic_stats: dict with 'mean' and 'std' per VGG layer
        
        Returns:
            styled: (B, 3, H, W)
        """
        _, _, _, real_feat = self.encode(real_images)
        
        B, C = real_feat.shape[:2]
        real_mean = real_feat.view(B, C, -1).mean(dim=2, keepdim=True)
        real_std = real_feat.view(B, C, -1).std(dim=2, keepdim=True) + 1e-8
        
        syn_mean = synthetic_stats['mean'].to(real_feat.device)
        syn_std = synthetic_stats['std'].to(real_feat.device)
        
        # AdaIN with pre-computed stats
        normalized = (real_feat.view(B, C, -1) - real_mean) / real_std
        stylized = normalized * syn_std + syn_mean
        
        styled_feat = stylized.view_as(real_feat)
        styled = self.decoder(styled_feat)
        
        return styled


def compute_synthetic_statistics(model, dataloader, device='cuda'):
    """
    Pre-compute mean/std of VGG features on synthetic dataset.
    Used for AdaIN style transfer at test time.
    """
    adain = AdaINStyleTransfer().to(device).eval()
    
    all_means = []
    all_stds = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            _, _, _, feat = adain.encode(images)
            B, C = feat.shape[:2]
            mean = feat.view(B, C, -1).mean(dim=2)  # (B, C)
            std = feat.view(B, C, -1).std(dim=2)     # (B, C)
            all_means.append(mean.cpu())
            all_stds.append(std.cpu())
    
    global_mean = torch.cat(all_means).mean(dim=0, keepdim=True).unsqueeze(-1)  # (1, C, 1)
    global_std = torch.cat(all_stds).mean(dim=0, keepdim=True).unsqueeze(-1)    # (1, C, 1)
    
    return {'mean': global_mean, 'std': global_std}
