"""
Training script for Hybrid CNN-Transformer 3D Reconstruction.
FIXED VERSION — Key changes:
  1. Differential learning rates (encoder lr << decoder lr)
  2. Proper warmup using SequentialLR (no manual override conflict)
  3. Better logging

Usage:
  python train.py --config configs/config.yaml
  python train.py --config configs/config.yaml --resume checkpoints/best.pt
  python train.py --config configs/config.yaml --dann   # enable DANN
"""

import os
import sys
import time
import argparse
import yaml
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from model import HybridReconstructor, DomainDiscriminator, build_model
from dataset import create_dataloaders, get_val_transform, GSODataset
from losses import ChamferLoss, evaluate_reconstruction, dann_loss, compute_dann_lambda


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(model, train_loader, optimizer, criterion, scaler, device,
                    epoch, cfg, discriminator=None, disc_optimizer=None,
                    real_loader_iter=None):
    """Train for one epoch."""
    model.train()
    if discriminator:
        discriminator.train()
    
    loss_meter = AverageMeter()
    cd_meter = AverageMeter()
    dann_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] CD:{postfix}')

    for batch_idx, (images, gt_points, _) in enumerate(pbar):
        images = images.to(device)
        gt_points = gt_points.to(device)
        B = images.shape[0]

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=cfg['training'].get('mixed_precision', True)):
            # Forward pass
            pred_points = model(images)

            # Chamfer loss
            recon_loss = criterion(pred_points, gt_points)
            total_loss = recon_loss

            # DANN loss (Strategy 3)
            domain_loss_val = 0.0
            if discriminator and real_loader_iter:
                dann_cfg = cfg.get('domain_adaptation', {})
                lambda_ = compute_dann_lambda(
                    epoch, cfg['training']['num_epochs'],
                    alpha=dann_cfg.get('dann_alpha', 10.0)
                )

                # Get real images
                try:
                    real_images, _, _ = next(real_loader_iter)
                except StopIteration:
                    real_images = images  # fallback

                real_images = real_images.to(device)

                # Encode both domains
                synth_features = model.encode(images)
                real_features = model.encode(real_images)

                # Domain labels: 0 = synthetic, 1 = real
                synth_labels = torch.zeros(B, 1, device=device)
                real_labels = torch.ones(real_images.shape[0], 1, device=device)

                # Domain predictions
                synth_logits = discriminator(synth_features, lambda_)
                real_logits = discriminator(real_features, lambda_)

                d_loss = (dann_loss(synth_logits, synth_labels) +
                         dann_loss(real_logits, real_labels)) / 2

                total_loss = total_loss + dann_cfg.get('dann_lambda', 1.0) * d_loss
                domain_loss_val = d_loss.item()

        # Backward
        scaler.scale(total_loss).backward()

        # Gradient clipping
        if cfg['training'].get('grad_clip', 0) > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
            if discriminator:
                nn.utils.clip_grad_norm_(discriminator.parameters(), cfg['training']['grad_clip'])

        scaler.step(optimizer)
        if disc_optimizer:
            scaler.step(disc_optimizer)
        scaler.update()

        # Update meters
        loss_meter.update(total_loss.item(), B)
        cd_meter.update(recon_loss.item(), B)
        if domain_loss_val > 0:
            dann_meter.update(domain_loss_val, B)

        # Update progress bar
        pbar.set_postfix_str(f"{cd_meter.avg:.6f}")
    
    return {
        'train_loss': loss_meter.avg,
        'train_cd': cd_meter.avg,
        'train_dann': dann_meter.avg if discriminator else 0.0,
    }


@torch.no_grad()
def evaluate(model, val_loader, criterion, device, thresholds=[0.01, 0.02, 0.05]):
    """Evaluate on validation/test set."""
    model.eval()
    
    loss_meter = AverageMeter()
    all_metrics = defaultdict(list)
    
    pbar = tqdm(val_loader, desc="Validating", leave=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    for images, gt_points, _ in pbar:
        images = images.to(device)
        gt_points = gt_points.to(device)
        B = images.shape[0]
        
        pred_points = model(images)
        loss = criterion(pred_points, gt_points)
        loss_meter.update(loss.item(), B)
        
        # Detailed metrics
        metrics = evaluate_reconstruction(pred_points, gt_points, thresholds)
        for k, v in metrics.items():
            all_metrics[k].append(v)
    
    # Average
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    avg_metrics['val_loss'] = loss_meter.avg
    
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Train 3D Reconstruction Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--dann', action='store_true', help='Enable DANN domain adaptation')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='Train from scratch (disable pretrained weights)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # Override settings
    if args.dann:
        cfg['domain_adaptation']['dann_enabled'] = True
    if args.no_pretrained:
        cfg['model']['use_pretrained'] = False
    
    # Setup
    set_seed(cfg['training']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create directories
    os.makedirs(cfg['logging']['save_dir'], exist_ok=True)
    os.makedirs(cfg['logging']['log_dir'], exist_ok=True)
    
    # Build model
    model, discriminator = build_model(cfg)
    model = model.to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")
    
    if discriminator:
        discriminator = discriminator.to(device)
        print(f"Discriminator parameters: {count_parameters(discriminator):,}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(cfg)
    
    # Real data loader for DANN
    real_loader_iter = None
    if discriminator:
        gso_dataset = GSODataset(
            cfg['data']['gso_root'],
            num_points=cfg['data']['num_points'],
            transform=get_val_transform(cfg['augmentation']),
        )
        if len(gso_dataset) > 0:
            from torch.utils.data import DataLoader
            real_loader = DataLoader(
                gso_dataset,
                batch_size=cfg['training']['batch_size'],
                shuffle=True, num_workers=2, drop_last=True,
            )
            real_loader_iter = iter(real_loader)
    
    # Loss
    criterion = ChamferLoss()
    
    # ─── FIX: Differential Learning Rates ─────────────────────
    # Pretrained encoder needs LOWER lr to preserve ImageNet features.
    # Randomly-initialized decoder needs HIGHER lr to learn quickly.
    base_lr = cfg['training']['learning_rate']  # 1e-4
    
    encoder_params = list(model.encoder.parameters())
    encoder_param_ids = set(id(p) for p in encoder_params)
    decoder_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]
    
    encoder_lr = base_lr * 0.2   # 2e-5 — gentle fine-tuning
    decoder_lr = base_lr * 5.0   # 5e-4 — fast learning for new layers
    
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': encoder_lr, 'name': 'encoder'},
        {'params': decoder_params, 'lr': decoder_lr, 'name': 'decoder'},
    ], weight_decay=cfg['training']['weight_decay'])
    
    print(f"  Encoder LR: {encoder_lr:.2e}")
    print(f"  Decoder LR: {decoder_lr:.2e}")
    print(f"  Encoder params: {sum(p.numel() for p in encoder_params):,}")
    print(f"  Decoder params: {sum(p.numel() for p in decoder_params):,}")
    # ──────────────────────────────────────────────────────────
    
    disc_optimizer = None
    if discriminator:
        disc_optimizer = optim.Adam(
            discriminator.parameters(), lr=base_lr
        )
    
    # ─── FIX: Proper Warmup + Cosine Scheduler ───────────────
    warmup_epochs = cfg['training'].get('warmup_epochs', 5)
    num_epochs = cfg['training']['num_epochs']
    
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs,
        eta_min=base_lr * 0.01,
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    # ──────────────────────────────────────────────────────────
    
    # Mixed precision
    scaler = GradScaler(enabled=cfg['training'].get('mixed_precision', True))
    
    # Resume
    start_epoch = 0
    best_val_cd = float('inf')
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        # Note: optimizer state may not match new param groups — skip loading it
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_cd = ckpt.get('best_val_cd', float('inf'))
        print(f"Resumed model from epoch {start_epoch}, best CD: {best_val_cd:.6f}")
        print(f"  (Optimizer re-initialized with new param groups)")
        # Fast-forward scheduler to match resumed epoch so LR is correct
        for _ in range(start_epoch):
            scheduler.step()
    
    # ─── Training Loop ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("Starting Training")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {cfg['training']['batch_size']}")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print("=" * 60)
    
    epoch_times = []
    training_start = time.time()
    log_file = os.path.join(cfg['logging']['log_dir'], 'training_log.txt')
    # Append mode so resumed runs continue the log
    log_fh = open(log_file, 'a')
    log_fh.write(f"\n{'='*60}\n")
    log_fh.write(f"Resumed training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_fh.write(f"Epochs: {start_epoch+1} -> {num_epochs}\n")
    log_fh.write(f"{'='*60}\n")
    log_fh.flush()

    epoch_pbar = tqdm(range(start_epoch, num_epochs), desc="Overall",
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} epochs [{elapsed}<{remaining}]',
                      position=0)
    for epoch in epoch_pbar:
        epoch_start = time.time()

        # Print current LR for both groups
        enc_lr = optimizer.param_groups[0]['lr']
        dec_lr = optimizer.param_groups[1]['lr']
        epoch_pbar.set_description(f"Overall (best CD: {best_val_cd:.6f})")
        tqdm.write(f"\nEpoch {epoch + 1}/{num_epochs} "
                   f"(enc_lr={enc_lr:.2e}, dec_lr={dec_lr:.2e})")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            epoch, cfg, discriminator, disc_optimizer, real_loader_iter,
        )
        
        # Step scheduler (handles warmup + cosine automatically)
        scheduler.step()
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device,
                              cfg['evaluation']['fscore_thresholds'])
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = np.mean(epoch_times)
        remaining_epochs = num_epochs - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_str = str(timedelta(seconds=int(eta_seconds)))
        finish_time = datetime.now() + timedelta(seconds=eta_seconds)

        # Print results
        tqdm.write(f"  Train Loss: {train_metrics['train_loss']:.6f} | "
                   f"Train CD: {train_metrics['train_cd']:.6f}")
        tqdm.write(f"  Val Loss:   {val_metrics['val_loss']:.6f} | "
                   f"Val CD:   {val_metrics['chamfer_distance']:.6f}")
        for t in cfg['evaluation']['fscore_thresholds']:
            tqdm.write(f"  F-Score@{t}: {val_metrics.get(f'f_score@{t}', 0):.4f}")
        tqdm.write(f"  Epoch time: {epoch_time:.1f}s | "
                   f"ETA: {eta_str} | "
                   f"Est. finish: {finish_time.strftime('%b %d %I:%M %p')}")

        # Log to file
        log_fh.write(f"\nEpoch {epoch+1}/{num_epochs} "
                     f"(enc_lr={enc_lr:.2e}, dec_lr={dec_lr:.2e})\n")
        log_fh.write(f"  Train CD: {train_metrics['train_cd']:.6f} | "
                     f"Val CD: {val_metrics['chamfer_distance']:.6f}\n")
        for t in cfg['evaluation']['fscore_thresholds']:
            log_fh.write(f"  F-Score@{t}: {val_metrics.get(f'f_score@{t}', 0):.4f}\n")
        log_fh.write(f"  Time: {epoch_time:.1f}s | ETA: {eta_str} | "
                     f"Finish: {finish_time.strftime('%b %d %I:%M %p')}\n")
        log_fh.flush()

        # Save best
        is_best = val_metrics['chamfer_distance'] < best_val_cd
        if is_best:
            best_val_cd = val_metrics['chamfer_distance']
            save_path = os.path.join(cfg['logging']['save_dir'], 'best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_cd': best_val_cd,
                'val_metrics': val_metrics,
                'config': cfg,
            }, save_path)
            tqdm.write(f"  ** New best model saved! CD: {best_val_cd:.6f}")
            log_fh.write(f"  ** New best model! CD: {best_val_cd:.6f}\n")
            log_fh.flush()
        
        # Periodic save
        if (epoch + 1) % cfg['logging']['save_every'] == 0:
            save_path = os.path.join(cfg['logging']['save_dir'], f'epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_cd': best_val_cd,
                'config': cfg,
            }, save_path)
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best Val CD: {best_val_cd:.6f}")
    print("=" * 60)
    
    # ─── Final Test Evaluation ────────────────────────────────
    print("\nEvaluating on synthetic test set...")
    best_ckpt = torch.load(os.path.join(cfg['logging']['save_dir'], 'best.pt'),
                           map_location=device)
    model.load_state_dict(best_ckpt['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, criterion, device,
                           cfg['evaluation']['fscore_thresholds'])
    
    print("\n-- Synthetic Test Results --")
    print(f"  Chamfer Distance: {test_metrics['chamfer_distance']:.6f}")
    for t in cfg['evaluation']['fscore_thresholds']:
        print(f"  F-Score@{t}: {test_metrics.get(f'f_score@{t}', 0):.4f}")
    
    # Save final results
    import json
    results_path = os.path.join(cfg['logging']['log_dir'], 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'synthetic_test': test_metrics,
            'best_val_cd': best_val_cd,
            'total_params': model.num_params,
        }, f, indent=2)
    
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
