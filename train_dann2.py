"""
DANN Training for 2048-pt model — saves to visualizations2/dann/.

Reduced batch_size to 8 for 16GB VRAM with 2048 query points.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
from pathlib import Path

from config import config_from_yaml
from model import build_model
from dataset import ShapeNetCap3DDataset, get_train_transform, get_val_transform
from losses import chamfer_distance, ChamferDistanceLoss, f_score
from strategies import DomainDiscriminator, dann_lambda_schedule


class RealImageDataset(Dataset):
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


@torch.no_grad()
def evaluate(model, loader, device, verbose=False):
    model.eval()
    all_cd = []
    all_fs = {0.01: [], 0.02: [], 0.05: []}

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            img, pc = batch[0], batch[1]
        else:
            img, pc = batch["image"], batch["points"]
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


def main():
    import argparse
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

    # Load model
    model, _ = build_model(cfg_dict)
    model = model.to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint epoch {ckpt['epoch']} from {args.checkpoint}")

    # Domain discriminator
    query_dim = cfg_dict['model'].get('query_dim', 256)
    discriminator = DomainDiscriminator(input_dim=query_dim, hidden_dim=256).to(device)

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

    real_dataset = RealImageDataset(args.real_images, transform=val_transform)

    bs = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=4)
    real_loader = DataLoader(real_dataset, batch_size=bs, shuffle=True, num_workers=2, drop_last=True)

    # Optimizers — differential LR
    optimizer = optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 1e-5},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('encoder')], 'lr': 5e-5},
        {'params': discriminator.parameters(), 'lr': 1e-4},
    ], weight_decay=1e-4)

    recon_criterion = ChamferDistanceLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    ckpt_dir = "./checkpoints/dann_2048"
    log_dir = "./visualizations2/dann"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Log file
    log_file = os.path.join(log_dir, "dann_training_log.txt")
    logf = open(log_file, 'w')

    best_val_cd = float('inf')
    patience = 0

    # Baseline val CD before DANN
    print("Evaluating baseline (pre-DANN)...")
    baseline_cd = evaluate(model, val_loader, device, verbose=True)

    msg = f"DANN Training: {args.epochs} epochs, lambda={args.dann_lambda}, batch_size={bs}\n"
    msg += f"  Synthetic train: {len(train_dataset)} | Real: {len(real_dataset)}\n"
    msg += f"  Baseline val CD: {baseline_cd:.6f}\n"
    print(msg)
    logf.write(msg + "\n"); logf.flush()

    for epoch in range(args.epochs):
        model.train()
        discriminator.train()

        total_recon_loss = 0
        total_domain_loss = 0
        n_batches = 0

        lambda_val = args.dann_lambda * dann_lambda_schedule(epoch, args.epochs)
        discriminator.set_lambda(lambda_val)

        real_iter = iter(real_loader)

        pbar = tqdm(train_loader, desc=f"DANN {epoch+1}/{args.epochs}")
        for batch in pbar:
            if isinstance(batch, (list, tuple)):
                img, pc = batch[0], batch[1]
            else:
                img, pc = batch["image"], batch["points"]
            img = img.to(device)
            pc = pc.to(device)

            try:
                real_img = next(real_iter).to(device)
            except StopIteration:
                real_iter = iter(real_loader)
                real_img = next(real_iter).to(device)

            min_bs = min(img.shape[0], real_img.shape[0])
            img = img[:min_bs]
            pc = pc[:min_bs]
            real_img = real_img[:min_bs]

            # Reconstruction
            pred_points = model(img)
            recon_loss = recon_criterion(pred_points, pc)

            # Domain discrimination
            syn_features = model.encode(img)
            real_features = model.encode(real_img)

            syn_domain = discriminator(syn_features)
            real_domain = discriminator(real_features)

            syn_labels = torch.zeros_like(syn_domain)
            real_labels = torch.ones_like(real_domain)

            domain_loss = domain_criterion(syn_domain, syn_labels) + \
                          domain_criterion(real_domain, real_labels)

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

        val_cd = evaluate(model, val_loader, device)

        msg = (f"Epoch {epoch+1}/{args.epochs}: recon={avg_recon:.5f}, domain={avg_domain:.4f}, "
               f"val_CD={val_cd:.6f}, lambda={lambda_val:.4f}")
        print(f"  {msg}")
        logf.write(msg + "\n"); logf.flush()

        if val_cd < best_val_cd:
            best_val_cd = val_cd
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'val_cd': val_cd,
                'baseline_cd': baseline_cd,
            }, os.path.join(ckpt_dir, 'best.pt'))
            msg = f"  ** New best: {val_cd:.6f} (saved)"
            print(msg)
            logf.write(msg + "\n"); logf.flush()
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                msg = f"  Early stopping at epoch {epoch+1}"
                print(msg)
                logf.write(msg + "\n"); logf.flush()
                break

    # Final eval with best model
    best_ckpt = torch.load(os.path.join(ckpt_dir, 'best.pt'), map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])

    print("\n=== Final DANN Evaluation ===")
    logf.write("\n=== Final DANN Evaluation ===\n")
    final_cd = evaluate(model, val_loader, device, verbose=True)

    summary = (f"\nDANN Summary:\n"
               f"  Baseline val CD: {baseline_cd:.6f}\n"
               f"  Best DANN val CD: {best_val_cd:.6f}\n"
               f"  Improvement: {(baseline_cd - best_val_cd) / baseline_cd * 100:.2f}%\n")
    print(summary)
    logf.write(summary)
    logf.close()

    print(f"Log saved to {log_file}")
    print(f"Best checkpoint saved to {ckpt_dir}/best.pt")


if __name__ == "__main__":
    main()
