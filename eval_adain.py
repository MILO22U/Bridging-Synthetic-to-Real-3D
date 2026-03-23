"""
AdaIN Style Transfer Evaluation (Strategy 4).

No training needed — applies style transfer at test time to make
real images look more synthetic before feeding to model.

Usage:
    python eval_adain.py --config config.yaml
"""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import config_from_yaml
from model import build_model
from dataset import ShapeNetCap3DDataset, get_val_transform
from losses import chamfer_distance, f_score
from strategies import AdaINStyleTransfer


@torch.no_grad()
def compute_synthetic_stats(model_adain, dataset, device, num_samples=200):
    """Compute mean/std of VGG features on synthetic images."""
    print("Computing synthetic feature statistics...")
    all_means = []
    all_stds = []

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    for idx in tqdm(indices, desc="Computing stats"):
        img, _, _ = dataset[idx]
        img = img.unsqueeze(0).to(device)
        _, _, _, feat = model_adain.encode(img)
        B, C = feat.shape[:2]
        mean = feat.view(B, C, -1).mean(dim=2)
        std = feat.view(B, C, -1).std(dim=2)
        all_means.append(mean.cpu())
        all_stds.append(std.cpu())

    global_mean = torch.cat(all_means).mean(dim=0, keepdim=True).unsqueeze(-1)
    global_std = torch.cat(all_stds).mean(dim=0, keepdim=True).unsqueeze(-1)
    print(f"  Stats computed from {len(indices)} samples")
    return {'mean': global_mean, 'std': global_std}


@torch.no_grad()
def evaluate_with_adain(model, adain_model, real_photos_dir, syn_stats, transform, device, output_dir):
    """Run inference on real photos with AdaIN preprocessing."""
    model.eval()
    adain_model.eval()

    real_images = sorted(
        list(Path(real_photos_dir).glob("*.jpg")) +
        list(Path(real_photos_dir).glob("*.png")) +
        list(Path(real_photos_dir).glob("*.jpeg"))
    )
    print(f"\nEvaluating {len(real_images)} real images with AdaIN...")

    os.makedirs(output_dir, exist_ok=True)

    for img_path in real_images:
        pil_img = Image.open(str(img_path)).convert("RGB")
        img_tensor = transform(pil_img).unsqueeze(0).to(device)

        # Original prediction (no AdaIN)
        pred_orig = model(img_tensor)[0].cpu().numpy()

        # AdaIN: transform real → synthetic style
        styled = adain_model.transfer_to_synthetic_style(img_tensor, syn_stats)
        styled = styled.clamp(0, 1)
        styled = torch.nn.functional.interpolate(styled, size=(224, 224), mode='bilinear', align_corners=False)

        # Re-normalize styled image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        styled_norm = (styled - mean) / std

        # Prediction on styled image
        pred_adain = model(styled_norm)[0].cpu().numpy()

        # Save comparison
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Original image
        img_np = img_tensor[0].cpu() * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + \
                 torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        axes[0].imshow(img_np.clamp(0,1).permute(1,2,0).numpy())
        axes[0].set_title("Real Input")
        axes[0].axis('off')

        # Styled image
        axes[1].imshow(styled[0].cpu().permute(1,2,0).clamp(0,1).numpy())
        axes[1].set_title("AdaIN Styled")
        axes[1].axis('off')

        # Original prediction
        ax2 = fig.add_subplot(1, 4, 3, projection='3d')
        ax2.scatter(pred_orig[:, 0], pred_orig[:, 1], pred_orig[:, 2], s=0.5, alpha=0.6)
        ax2.set_title("Pred (Original)")
        ax2.view_init(30, 45)

        # AdaIN prediction
        ax3 = fig.add_subplot(1, 4, 4, projection='3d')
        ax3.scatter(pred_adain[:, 0], pred_adain[:, 1], pred_adain[:, 2], s=0.5, alpha=0.6)
        ax3.set_title("Pred (AdaIN)")
        ax3.view_init(30, 45)

        plt.suptitle(f"AdaIN: {img_path.name}", fontsize=12)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"adain_{img_path.stem}.png")
        plt.savefig(save_path, dpi=100)
        plt.close()
        print(f"  Saved: {save_path}")


@torch.no_grad()
def evaluate_synthetic_with_adain(model, adain_model, test_dataset, syn_stats, device):
    """Evaluate on synthetic test set with and without AdaIN for comparison."""
    model.eval()
    adain_model.eval()

    print("\nEvaluating synthetic test set (AdaIN comparison)...")
    all_cd_orig = []
    all_cd_adain = []

    loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
    std_t = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)

    for img, pc, _ in tqdm(loader, desc="AdaIN eval"):
        img = img.to(device)
        pc = pc.to(device)

        # Original
        pred_orig = model(img)

        # AdaIN styled
        styled = adain_model.transfer_to_synthetic_style(img, syn_stats).clamp(0, 1)
        styled = torch.nn.functional.interpolate(styled, size=(224, 224), mode='bilinear', align_corners=False)
        styled_norm = (styled - mean_t) / std_t
        pred_adain = model(styled_norm)

        cd_o, _, _ = chamfer_distance(pred_orig, pc)
        cd_a, _, _ = chamfer_distance(pred_adain, pc)
        all_cd_orig.append(cd_o.item())
        all_cd_adain.append(cd_a.item())

        del img, pc, pred_orig, pred_adain, styled, styled_norm
        torch.cuda.empty_cache()

    print(f"\n  Original CD: {np.mean(all_cd_orig):.6f}")
    print(f"  AdaIN CD:    {np.mean(all_cd_adain):.6f}")
    return np.mean(all_cd_orig), np.mean(all_cd_adain)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_2048.yaml")
    parser.add_argument("--checkpoint", default="./checkpoints/retrain_2048/best.pt")
    parser.add_argument("--output", default="./visualizations2/adain_vgg")
    args = parser.parse_args()

    cfg = config_from_yaml(args.config)
    cfg_dict = cfg.to_dict()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load reconstruction model
    model, _ = build_model(cfg_dict)
    model = model.to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {args.checkpoint}")

    # Load AdaIN model
    adain_model = AdaINStyleTransfer().to(device).eval()
    print("AdaIN model loaded (VGG-19 encoder)")

    transform = get_val_transform(cfg_dict['augmentation'])

    # Synthetic dataset for stats
    train_dataset = ShapeNetCap3DDataset(
        shapenet_root=cfg.data.shapenet_root,
        cap3d_root=cfg.data.cap3d_root,
        categories=cfg.data.categories,
        split='train', num_points=cfg.data.num_points,
        num_views=cfg.data.num_views, transform=transform,
    )
    test_dataset = ShapeNetCap3DDataset(
        shapenet_root=cfg.data.shapenet_root,
        cap3d_root=cfg.data.cap3d_root,
        categories=cfg.data.categories,
        split='test', num_points=cfg.data.num_points,
        num_views=cfg.data.num_views, transform=transform,
    )

    # Compute synthetic feature statistics (use fewer samples to save VRAM)
    syn_stats = compute_synthetic_stats(adain_model, train_dataset, device, num_samples=100)
    torch.cuda.empty_cache()

    # Evaluate on synthetic test
    evaluate_synthetic_with_adain(model, adain_model, test_dataset, syn_stats, device)

    # Evaluate on real photos
    real_dir = cfg.data.real_photos_dir
    if os.path.exists(real_dir):
        evaluate_with_adain(model, adain_model, real_dir, syn_stats, transform, device, args.output)

    print("\nAdaIN evaluation complete!")


if __name__ == "__main__":
    main()
