"""
Hybrid CNN-Transformer for Single-Image 3D Point Cloud Reconstruction.

Architecture:
  1. ResNet-18/34 Image Encoder (pretrained ImageNet weights available)
  2. Cross-Attention Bridge (image tokens -> 3D query tokens)
  3. Transformer Decoder (self-attention refinement)
  4. MLP Head (query tokens -> xyz coordinates)

Updated per professor feedback: use_pretrained=True by default.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ──────────────────────────────────────────────────────────────
# 1. Image Encoder (ResNet backbone)
# ──────────────────────────────────────────────────────────────

class ResNetEncoder(nn.Module):
    """
    ResNet-18 or ResNet-34 encoder.
    Outputs a grid of feature vectors (B, num_tokens, encoder_dim).
    
    With pretrained=True (professor's recommendation), uses ImageNet weights.
    This dramatically improves convergence and feature quality.
    """
    
    def __init__(self, backbone='resnet18', pretrained=True, output_dim=512):
        super().__init__()
        
        if backbone == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet18(weights=weights)
            feat_dim = 512
        elif backbone == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet34(weights=weights)
            feat_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove avgpool and fc — keep spatial features
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        
        # Optional projection if output_dim differs
        self.proj = nn.Identity()
        if feat_dim != output_dim:
            self.proj = nn.Linear(feat_dim, output_dim)
        
        self.output_dim = output_dim
        
        if pretrained:
            print(f"[OK] Loaded pretrained {backbone} (ImageNet) encoder")
        else:
            print(f"[--] Training {backbone} encoder from scratch")
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224) input image
        Returns:
            (B, 49, output_dim) — 7×7 grid of feature vectors
        """
        feat = self.features(x)           # (B, 512, 7, 7)
        B, C, H, W = feat.shape
        feat = feat.flatten(2).transpose(1, 2)  # (B, 49, 512)
        feat = self.proj(feat)             # (B, 49, output_dim)
        return feat


# ──────────────────────────────────────────────────────────────
# 2. Positional Encoding
# ──────────────────────────────────────────────────────────────

class LearnedPositionalEncoding(nn.Module):
    """Learnable positional encoding for tokens."""
    
    def __init__(self, num_tokens, dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
    
    def forward(self, x):
        return x + self.pos_embed


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (non-learnable)."""
    
    def __init__(self, num_tokens, dim):
        super().__init__()
        pe = torch.zeros(num_tokens, dim)
        position = torch.arange(0, num_tokens).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe


# ──────────────────────────────────────────────────────────────
# 3. Cross-Attention Bridge
# ──────────────────────────────────────────────────────────────

class CrossAttentionLayer(nn.Module):
    """
    Cross-attention: queries attend to image features.
    Query tokens = future 3D points, Key/Value = image tokens.
    """
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm_kv = nn.LayerNorm(dim)
    
    def forward(self, queries, kv):
        """
        Args:
            queries: (B, N_q, D) — 3D query tokens
            kv: (B, N_kv, D) — image tokens
        Returns:
            (B, N_q, D) — updated query tokens
        """
        kv = self.norm_kv(kv)
        
        # Cross-attention
        residual = queries
        queries = self.norm1(queries)
        queries = residual + self.cross_attn(queries, kv, kv)[0]
        
        # FFN
        residual = queries
        queries = residual + self.ffn(self.norm2(queries))
        
        return queries


# ──────────────────────────────────────────────────────────────
# 4. Transformer Decoder (Self-Attention)
# ──────────────────────────────────────────────────────────────

class SelfAttentionLayer(nn.Module):
    """Standard transformer self-attention layer for point refinement."""
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, N, D) — point tokens
        Returns:
            (B, N, D) — refined tokens
        """
        residual = x
        x_norm = self.norm1(x)
        x = residual + self.self_attn(x_norm, x_norm, x_norm)[0]
        
        residual = x
        x = residual + self.ffn(self.norm2(x))
        
        return x


# ──────────────────────────────────────────────────────────────
# 5. MLP Head
# ──────────────────────────────────────────────────────────────

class PointHead(nn.Module):
    """MLP head: maps each query token to xyz coordinate."""
    
    def __init__(self, dim, hidden_dim=512, output_dim=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),  # output in [-1, 1] since point clouds are normalized
        )
    
    def forward(self, x):
        return self.mlp(x)


# ──────────────────────────────────────────────────────────────
# 6. Full Model: HybridReconstructor
# ──────────────────────────────────────────────────────────────

class HybridReconstructor(nn.Module):
    """
    Hybrid CNN-Transformer for single-image 3D point cloud reconstruction.
    
    Pipeline:
      Image -> ResNet Encoder -> Cross-Attention Bridge -> Transformer Decoder -> Point Cloud
    
    Args (from config):
      encoder_backbone: 'resnet18' or 'resnet34'
      use_pretrained: whether to use ImageNet pretrained weights
      encoder_dim: dimension of encoder output features
      num_query_tokens: number of output 3D points
      query_dim: dimension of query/decoder tokens
      cross_attn_layers: number of cross-attention layers
      self_attn_layers: number of self-attention layers
      ...
    """
    
    def __init__(self, cfg_model):
        super().__init__()
        
        encoder_dim = cfg_model['encoder_dim']
        query_dim = cfg_model['query_dim']
        num_queries = cfg_model['num_query_tokens']
        num_image_tokens = cfg_model['num_image_tokens']
        
        # 1. Image Encoder
        self.encoder = ResNetEncoder(
            backbone=cfg_model['encoder_backbone'],
            pretrained=cfg_model.get('use_pretrained', True),
            output_dim=encoder_dim,
        )
        
        # Project encoder features to query_dim if needed
        self.encoder_proj = nn.Identity()
        if encoder_dim != query_dim:
            self.encoder_proj = nn.Linear(encoder_dim, query_dim)
        
        # Positional encoding for image tokens
        self.image_pos = LearnedPositionalEncoding(num_image_tokens, query_dim)
        
        # 2. Learnable 3D query tokens
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, query_dim) * 0.02
        )
        self.query_pos = LearnedPositionalEncoding(num_queries, query_dim)
        
        # 3. Cross-Attention Bridge
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(
                dim=query_dim,
                num_heads=cfg_model['cross_attn_heads'],
                dropout=cfg_model['dropout'],
            )
            for _ in range(cfg_model['cross_attn_layers'])
        ])
        
        # 4. Self-Attention Decoder
        self.self_attn_layers = nn.ModuleList([
            SelfAttentionLayer(
                dim=query_dim,
                num_heads=cfg_model['self_attn_heads'],
                dropout=cfg_model['dropout'],
            )
            for _ in range(cfg_model['self_attn_layers'])
        ])
        
        # 5. Point Head
        self.point_head = PointHead(
            dim=query_dim,
            hidden_dim=cfg_model['mlp_hidden_dim'],
            output_dim=cfg_model['output_dim'],
        )
        
        self._num_params = sum(p.numel() for p in self.parameters())
        print(f"HybridReconstructor: {self._num_params / 1e6:.1f}M parameters")
    
    def encode(self, images):
        """Extract image features. Useful for domain adaptation."""
        feat = self.encoder(images)          # (B, 49, encoder_dim)
        feat = self.encoder_proj(feat)       # (B, 49, query_dim)
        feat = self.image_pos(feat)          # add positional encoding
        return feat
    
    def decode(self, image_features):
        """Decode image features to point cloud."""
        B = image_features.shape[0]
        
        # Expand query tokens for batch
        queries = self.query_tokens.expand(B, -1, -1)  # (B, N_q, D)
        queries = self.query_pos(queries)
        
        # Cross-attention
        for layer in self.cross_attn_layers:
            queries = layer(queries, image_features)
        
        # Self-attention
        for layer in self.self_attn_layers:
            queries = layer(queries)
        
        # Point prediction
        points = self.point_head(queries)  # (B, N_q, 3)
        return points
    
    def forward(self, images):
        """
        Args:
            images: (B, 3, 224, 224)
        Returns:
            points: (B, num_query_tokens, 3) — predicted 3D point cloud
        """
        features = self.encode(images)
        points = self.decode(features)
        return points
    
    @property
    def num_params(self):
        return self._num_params


# ──────────────────────────────────────────────────────────────
# 7. Domain Discriminator (for DANN — Strategy 3)
# ──────────────────────────────────────────────────────────────

class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial domain adaptation."""
    
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_):
        self.lambda_ = lambda_


class DomainDiscriminator(nn.Module):
    """
    Binary domain classifier: synthetic (0) vs real (1).
    Applied on pooled image features after the encoder.
    Uses gradient reversal to learn domain-invariant features.
    """
    
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, features, lambda_=1.0):
        """
        Args:
            features: (B, N_tokens, D) image features
            lambda_: gradient reversal strength
        Returns:
            domain_logits: (B, 1)
        """
        self.grl.set_lambda(lambda_)
        
        # Pool over spatial tokens
        pooled = features.mean(dim=1)  # (B, D)
        reversed_feat = self.grl(pooled)
        logits = self.classifier(reversed_feat)
        return logits


# ──────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────

def build_model(cfg):
    """Build model and optionally domain discriminator from config."""
    model = HybridReconstructor(cfg['model'])
    
    discriminator = None
    if cfg.get('domain_adaptation', {}).get('dann_enabled', False):
        discriminator = DomainDiscriminator(
            input_dim=cfg['model']['query_dim'],
            hidden_dim=cfg['domain_adaptation']['dann_discriminator_hidden'],
        )
        print("[OK] DANN Domain Discriminator enabled")
    
    return model, discriminator