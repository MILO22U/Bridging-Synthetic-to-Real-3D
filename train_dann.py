"""
DANN Training — Fine-tune model with Domain Adversarial Neural Network.

Uses real-world images (unlabeled) to make encoder domain-invariant.
Starts from best pre-trained checkpoint.

Usage:
    python train_dann.py --config config.yaml
"""

import os
import random
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from PIL import Image
from pathlib import Path

from config import config_from_yaml
from model import build_model
from dataset import ShapeNetCap3DDataset, get_train_transform, get_val_transform
from losses import chamfer_distance, ChamferDistanceLoss, f_score
from strategies import DomainDiscriminator, dann_lambda_schedule


class RealImageDataset(Dataset):
    """Simple dataset of real-world images (no labels needed for DANN)."""
    def __init__(self, image_dir, transform=None):
        self.images = sorted(
            list(Path(image_dir).glob("*.jpg")) +
            list(Path(image_dir).glob("*.png")) +
            list(Path(image_dir).glob("*.jpeg"))
        )
        self.transform = transform
        print(f"RealImageDataset: {len(self.images)} images from {image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(str(self.images[idx])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_2048.yaml")
    parser.add_argument("--checkpoint", default="./checkpoints/retrain_2048/best.pt")
    parser.add_argument("--real-images", default="D:/DL/data/real_domain_images")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dann-lambda", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    cfg = config_from_yaml(args.config)
    cfg_dict = cfg.to_dict()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load model from best checkpoint
    model, _ = build_model(cfg_dict)
    model = model.to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Domain discriminator
    discriminator = DomainDiscriminator(input_dim=256, hidden_dim=256).to(device)

    # Datasets
    train_transform = get_train_transform(cfg_dict['augmentation'])
    val_transform = get_val_transform(cfg_dict['augmentation'])

    train_dataset = ShapeNetCap3DDataset(
        shapenet_root=cfg.data.shapenet_root,
        cap3d_root=cfg.data.cap3d_root,
        categories=cfg.data.categories,
        split='train', num_points=cfg.data.num_points,
        num_views=cfg.data.num_views, transform=train_transform,
    )
    val_dataset = ShapeNetCap3DDataset(
        shapenet_root=cfg.data.shapenet_root,
        cap3d_root=cfg.data.cap3d_root,
        categories=cfg.data.categories,
        split='val', num_points=cfg.data.num_points,
        num_views=cfg.data.num_views, transform=val_transform,
    )
    test_dataset = ShapeNetCap3DDataset(
        shapenet_root=cfg.data.shapenet_root,
        cap3d_root=cfg.data.cap3d_root,
        categories=cfg.data.categories,
        split='test', num_points=cfg.data.num_points,
        num_views=cfg.data.num_views, transform=val_transform,
    )

    real_dataset = RealImageDataset(args.real_images, transform=val_transform)

    bs = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=4)
    real_loader = DataLoader(real_dataset, batch_size=bs, shuffle=True, num_workers=2, drop_last=True)

    # Optimizers
    optimizer = optim.AdamW([
        {'params': model.parameters(), 'lr': 2e-5},
        {'params': discriminator.parameters(), 'lr': 1e-4},
    ], weight_decay=1e-4)

    recon_criterion = ChamferDistanceLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    ckpt_dir = "./checkpoints/dann_2048"
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_cd = float('inf')
    patience = 0

    print(f"\nDANN Training: {args.epochs} epochs, lambda={args.dann_lambda}")
    print(f"  Synthetic train: {len(train_dataset)} | Real: {len(real_dataset)}")

    for epoch in range(args.epochs):
        model.train()
        discriminator.train()

        total_recon_loss = 0
        total_domain_loss = 0
        n_batches = 0

        # Lambda schedule
        lambda_val = args.dann_lambda * dann_lambda_schedule(epoch, args.epochs)
        discriminator.set_lambda(lambda_val)

        real_iter = iter(real_loader)

        pbar = tqdm(train_loader, desc=f"DANN Epoch {epoch+1}/{args.epochs}")
        for img, pc, model_id in pbar:
            img = img.to(device)
            pc = pc.to(device)

            # Get real images (cycle if exhausted)
            try:
                real_img = next(real_iter).to(device)
            except StopIteration:
                real_iter = iter(real_loader)
                real_img = next(real_iter).to(device)

            # Match batch sizes
            min_bs = min(img.shape[0], real_img.shape[0])
            img = img[:min_bs]
            pc = pc[:min_bs]
            real_img = real_img[:min_bs]

            # Forward — reconstruction
            pred_points = model(img)
            recon_loss = recon_criterion(pred_points, pc)

            # Forward — domain discrimination
            syn_features = model.encode(img)
            real_features = model.encode(real_img)

            syn_domain = discriminator(syn_features)
            real_domain = discriminator(real_features)

            syn_labels = torch.zeros_like(syn_domain)
            real_labels = torch.ones_like(real_domain)

            domain_loss = domain_criterion(syn_domain, syn_labels) + \
                          domain_criterion(real_domain, real_labels)

            # Total loss
            total_loss = recon_loss + lambda_val * domain_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_recon_loss += recon_loss.item()
            total_domain_loss += domain_loss.item()
            n_batches += 1

            pbar.set_postfix(recon=f"{recon_loss.item():.4f}", dom=f"{domain_loss.item():.4f}")

        avg_recon = total_recon_loss / max(n_batches, 1)
        avg_domain = total_domain_loss / max(n_batches, 1)

        # Validate
        val_cd = evaluate(model, val_loader, device)
        print(f"  Epoch {epoch+1}: recon={avg_recon:.4f}, domain={avg_domain:.4f}, "
              f"val_CD={val_cd:.6f}, lambda={lambda_val:.3f}")

        if val_cd < best_val_cd:
            best_val_cd = val_cd
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_cd': val_cd}, os.path.join(ckpt_dir, 'best_model.pt'))
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Final eval
    ckpt = torch.load(os.path.join(ckpt_dir, 'best_model.pt'), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    test_cd = evaluate(model, test_loader, device, verbose=True)
    print(f"\nDANN Test CD: {test_cd:.6f}")


@torch.no_grad()
def evaluate(model, loader, device, verbose=False):
    model.eval()
    all_cd = []
    all_fs = {0.01: [], 0.02: [], 0.05: []}

    for img, pc, model_id in loader:
        img = img.to(device)
        pc = pc.to(device)
        pred = model(img)
        B = img.shape[0]
        for i in range(B):
            cd, _, _ = chamfer_distance(pred[i:i+1], pc[i:i+1])
            all_cd.append(cd.item())
            for t in [0.01, 0.02, 0.05]:
                fs = f_score(pred[i:i+1], pc[i:i+1], threshold=t)
                all_fs[t].append(fs.item())

    avg_cd = np.mean(all_cd)
    if verbose:
        print(f"  CD: {avg_cd:.6f} | F@0.01: {np.mean(all_fs[0.01]):.4f} | "
              f"F@0.02: {np.mean(all_fs[0.02]):.4f} | F@0.05: {np.mean(all_fs[0.05]):.4f}")
    return avg_cd


if __name__ == "__main__":
    main()
