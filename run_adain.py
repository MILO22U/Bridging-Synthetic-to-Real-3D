"""
AdaIN Style Transfer Experiment (Strategy 4)

Two approaches:
1. SimpleAdaIN: Match pixel-level channel statistics (no decoder needed)
2. Feature-level: Match VGG feature statistics

Compares inference on original vs styled real photos.
"""
import argparse
import yaml
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import HybridReconstructor
from dataset import create_dataloaders


def load_real_photos(photo_dir, image_size=224):
    """Load all real photos from directory."""
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_vis = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])
    
    photos = []
    names = []
    photos_vis = []
    
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    for f in sorted(os.listdir(photo_dir)):
        ext = os.path.splitext(f)[1].lower()
        if ext not in valid_ext:
            continue
        try:
            img = Image.open(os.path.join(photo_dir, f)).convert('RGB')
            photos.append(transform(img))
            photos_vis.append(transform_vis(img))
            names.append(os.path.splitext(f)[0])
        except Exception as e:
            print(f"  Skipping {f}: {e}")
    
    return torch.stack(photos), torch.stack(photos_vis), names


@torch.no_grad()
def compute_synthetic_pixel_stats(dataloader, max_batches=100):
    """Compute channel-wise mean/std of synthetic images (in normalized space).
    Uses Welford's online algorithm to avoid storing all images in memory."""
    print("Computing synthetic image statistics...")
    n = 0
    mean_acc = torch.zeros(1, 3, 1, 1)
    m2_acc = torch.zeros(1, 3, 1, 1)

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        images = batch[0]  # (B, 3, H, W) already normalized
        batch_mean = images.mean(dim=[0, 2, 3], keepdim=True)
        batch_var = images.var(dim=[0, 2, 3], keepdim=True)
        batch_n = images.shape[0]

        if n == 0:
            mean_acc = batch_mean
            m2_acc = batch_var * batch_n
        else:
            delta = batch_mean - mean_acc
            total = n + batch_n
            mean_acc = mean_acc + delta * batch_n / total
            m2_acc = m2_acc + batch_var * batch_n + delta ** 2 * n * batch_n / total
        n += batch_n

    syn_mean = mean_acc
    syn_std = torch.sqrt(m2_acc / n) + 1e-8

    print(f"  Computed from {n} images")
    return syn_mean, syn_std


def simple_adain(real_images, syn_mean, syn_std, alpha=0.5):
    """
    Simple AdaIN: match channel statistics of real images to synthetic.
    
    For each channel: x_styled = alpha * (sigma_syn/sigma_real * (x - mu_real) + mu_syn) + (1-alpha) * x
    """
    # Per-image channel stats
    B = real_images.shape[0]
    real_mean = real_images.mean(dim=[2, 3], keepdim=True)  # (B, 3, 1, 1)
    real_std = real_images.std(dim=[2, 3], keepdim=True) + 1e-8
    
    # Normalize then apply synthetic stats
    normalized = (real_images - real_mean) / real_std
    styled = normalized * syn_std + syn_mean
    
    # Interpolate
    output = alpha * styled + (1 - alpha) * real_images
    return output


@torch.no_grad()
def compute_synthetic_vgg_stats(dataloader, device, max_batches=50):
    """Compute VGG feature statistics from synthetic images."""
    print("Computing VGG feature statistics from synthetic data...")
    
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
    # Extract up to relu4_1 (layer index 21)
    encoder = vgg[:21].to(device).eval()
    
    all_means = []
    all_stds = []
    
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        images = batch[0].to(device)
        feat = encoder(images)
        B, C = feat.shape[:2]
        mean = feat.view(B, C, -1).mean(dim=2)
        std = feat.view(B, C, -1).std(dim=2) + 1e-8
        all_means.append(mean.cpu())
        all_stds.append(std.cpu())
        if (i + 1) % 10 == 0:
            print(f"  Processed {(i+1) * images.shape[0]} images...")
    
    global_mean = torch.cat(all_means).mean(dim=0)  # (C,)
    global_std = torch.cat(all_stds).mean(dim=0)
    
    del encoder
    torch.cuda.empty_cache()
    
    print(f"  Done! Stats from {len(all_means) * all_means[0].shape[0]} images")
    return global_mean, global_std


@torch.no_grad()
def vgg_adain(real_images, syn_mean, syn_std, device, alpha=0.5):
    """
    VGG feature-level AdaIN.
    
    Since we don't have a trained decoder, we modify the image in feature space
    and use the difference to adjust the pixel-level image.
    This is a practical approximation.
    """
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
    encoder = vgg[:21].to(device).eval()
    
    styled_images = []
    syn_mean = syn_mean.to(device)
    syn_std = syn_std.to(device)
    
    for i in range(real_images.shape[0]):
        img = real_images[i:i+1].to(device)
        feat = encoder(img)
        B, C = feat.shape[:2]
        
        real_mean = feat.view(B, C, -1).mean(dim=2)
        real_std = feat.view(B, C, -1).std(dim=2) + 1e-8
        
        # Compute color shift from feature space difference
        mean_shift = (syn_mean - real_mean).mean()
        std_ratio = (syn_std / real_std).mean()
        
        # Apply approximate correction in pixel space
        img_adjusted = (img - img.mean()) * std_ratio + img.mean() + mean_shift * 0.1
        img_styled = alpha * img_adjusted + (1 - alpha) * img
        styled_images.append(img_styled.cpu())
    
    del encoder
    torch.cuda.empty_cache()
    
    return torch.cat(styled_images)


def plot_point_cloud(ax, points, color='steelblue', title=''):
    """Plot 3D point cloud."""
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=color, s=0.5, alpha=0.6)
    ax.set_title(title, fontsize=9)
    max_range = max(np.max(np.abs(points)) * 1.1, 0.5)
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    ax.view_init(elev=30, azim=45)


def denormalize(tensor):
    """Denormalize for display."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--photo_dir', default='./data/real_photos')
    parser.add_argument('--save_dir', default='./visualizations/adain_experiment')
    parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    import gc

    print("=" * 60)
    print("AdaIN Style Transfer Experiment (Strategy 4)")
    print("=" * 60)

    # 1. Compute synthetic stats FIRST on CPU (no model on GPU yet)
    print("\nStep 1: Computing synthetic pixel stats (CPU only)...")
    stats_cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in cfg.items()}
    stats_cfg['training'] = dict(cfg['training'])
    stats_cfg['training']['batch_size'] = 4
    stats_cfg['training']['num_workers'] = 0  # no worker processes
    train_loader, _, _ = create_dataloaders(stats_cfg)
    syn_mean, syn_std = compute_synthetic_pixel_stats(train_loader, max_batches=50)
    del train_loader, stats_cfg
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Load real photos and apply AdaIN on CPU
    print(f"\nStep 2: Loading real photos from {args.photo_dir}...")
    real_images, real_vis, names = load_real_photos(args.photo_dir)
    print(f"  Found {len(names)} photos")

    print(f"\nStep 3: Applying AdaIN style transfer (alpha={args.alpha})...")
    styled_images = simple_adain(real_images, syn_mean, syn_std, alpha=args.alpha)
    del syn_mean, syn_std
    gc.collect()

    # 3. NOW load model on GPU for inference
    print("\nStep 4: Loading reconstruction model...")
    model = HybridReconstructor(cfg['model']).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Loaded epoch {ckpt['epoch']}")
    del ckpt
    gc.collect()
    torch.cuda.empty_cache()

    # 4. Run inference ONE image at a time
    print("\nStep 5: Running inference (1 image at a time)...")
    all_pred_original = []
    all_pred_styled = []

    for i in range(len(names)):
        with torch.no_grad():
            img_orig = real_images[i:i+1].to(device)
            pred_orig = model(img_orig).cpu()
            all_pred_original.append(pred_orig)
            del img_orig, pred_orig

            img_styled = styled_images[i:i+1].to(device)
            pred_styled = model(img_styled).cpu()
            all_pred_styled.append(pred_styled)
            del img_styled, pred_styled

        gc.collect()
        torch.cuda.empty_cache()
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(names)}]")

    all_pred_original = torch.cat(all_pred_original)
    all_pred_styled = torch.cat(all_pred_styled)

    # 6. Save visualizations
    print("\nSaving visualizations...")
    for i in range(len(names)):
        fig = plt.figure(figsize=(20, 5))
        
        # Original photo
        ax1 = fig.add_subplot(141)
        ax1.imshow(real_vis[i].permute(1, 2, 0).numpy())
        ax1.set_title('Original Photo', fontsize=10)
        ax1.axis('off')
        
        # Styled photo
        ax2 = fig.add_subplot(142)
        styled_vis = denormalize(styled_images[i])
        ax2.imshow(styled_vis.permute(1, 2, 0).numpy())
        ax2.set_title(f'AdaIN Styled (α={args.alpha})', fontsize=10)
        ax2.axis('off')
        
        # Predicted from original
        ax3 = fig.add_subplot(143, projection='3d')
        plot_point_cloud(ax3, all_pred_original[i], color='steelblue', title='Pred (Original)')
        
        # Predicted from styled
        ax4 = fig.add_subplot(144, projection='3d')
        plot_point_cloud(ax4, all_pred_styled[i], color='coral', title='Pred (AdaIN)')
        
        plt.suptitle(f'{names[i]}', fontsize=12)
        plt.tight_layout()
        save_path = os.path.join(args.save_dir, f'adain_{names[i]}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [{i+1}/{len(names)}] {names[i]}")

    print(f"\nDone! {len(names)} comparisons saved to {args.save_dir}/")
    print("Compare original vs styled predictions to see if AdaIN helps!")

if __name__ == '__main__':
    main()
