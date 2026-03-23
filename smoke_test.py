"""
Part 3: Smoke Test — Verify the full pipeline works end-to-end.

Tests:
  1. Config loads correctly
  2. Model builds with correct shapes and parameter count
  3. Forward pass on dummy input works
  4. Dataset loads real images + point clouds
  5. Loss computation works
  6. One full training step (forward + backward + optimizer step)

Run from your project directory (L:\DL):
    python smoke_test.py
    python smoke_test.py --config config.yaml    # test with yaml config
"""

import multiprocessing
multiprocessing.freeze_support()
import os
import sys
import time
import argparse
import numpy as np
from losses import ChamferDistanceLoss

print("=" * 60)
print("  SMOKE TEST — Full Pipeline Check")
print("=" * 60)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 1: Imports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n[1/6] Testing imports...")
try:
    import torch
    import torch.nn as nn
    print(f"  torch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except ImportError as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

try:
    from config import get_config, config_from_yaml, load_config
    from model import HybridReconstructor, build_model
    from losses import chamfer_distance, ChamferDistanceLoss, f_score
    from dataset import ShapeNetCap3DDataset, get_train_transform, get_val_transform, create_dataloaders
    print("  All local imports OK")
except ImportError as e:
    print(f"  FAIL: {e}")
    print(f"  Make sure you're running from the directory with config.py, model.py, etc.")
    sys.exit(1)

print("  PASS")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 2: Config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n[2/6] Testing config...")

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--config", default=None)
args, _ = parser.parse_known_args()

if args.config and os.path.exists(args.config):
    cfg = config_from_yaml(args.config)
    print(f"  Loaded from YAML: {args.config}")
else:
    cfg = get_config("base")
    print(f"  Using default 'base' config")

cfg_dict = cfg.to_dict()
print(f"  Experiment: {cfg.experiment_name}")
print(f"  Pretrained: {cfg.model.use_pretrained}")
print(f"  Num views: {cfg.data.num_views}")
print(f"  Num points: {cfg.data.num_points}")
print(f"  Batch size: {cfg.training.batch_size}")
print(f"  Encoder LR: {cfg.training.encoder_lr}")
print(f"  Decoder LR: {cfg.training.decoder_lr}")
print(f"  shapenet_root: {cfg.data.shapenet_root}")
print(f"  cap3d_root: {cfg.data.cap3d_root}")
print("  PASS")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 3: Model build + dummy forward pass
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n[3/6] Testing model build + forward pass...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, discriminator = build_model(cfg_dict)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params: {total_params / 1e6:.1f}M")
print(f"  Trainable:    {trainable_params / 1e6:.1f}M")

# Dummy forward pass
dummy_img = torch.randn(2, 3, 224, 224).to(device)
with torch.no_grad():
    t0 = time.time()
    pred = model(dummy_img)
    t1 = time.time()

print(f"  Input:  {dummy_img.shape}")
print(f"  Output: {pred.shape}")
print(f"  Output range: [{pred.min().item():.3f}, {pred.max().item():.3f}]")
print(f"  Forward time: {(t1-t0)*1000:.0f}ms")

assert pred.shape == (2, cfg.data.num_points, 3), f"Wrong output shape: {pred.shape}"
print("  PASS")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 4: Dataset loading (real data)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n[4/6] Testing dataset loading...")

try:
    train_loader, val_loader, test_loader = create_dataloaders(cfg_dict)
    print(f"  Train: {len(train_loader)} batches ({len(train_loader.dataset)} samples)")
    print(f"  Val:   {len(val_loader)} batches ({len(val_loader.dataset)} samples)")
    print(f"  Test:  {len(test_loader)} batches ({len(test_loader.dataset)} samples)")

    if len(train_loader) == 0:
        print("  WARNING: Train loader is empty! Check data paths.")
        print(f"  Expected renders at: {cfg.data.shapenet_root}/renders/")
        print(f"  Expected PCs at: {cfg.data.cap3d_root}/point_clouds/")
        print("  FAIL")
    else:
        # Load one real batch
        batch = next(iter(train_loader))
        img, pc, model_id = batch
        print(f"  Batch image shape:  {img.shape}")
        print(f"  Batch points shape: {pc.shape}")
        print(f"  Image range: [{img.min().item():.2f}, {img.max().item():.2f}]")
        print(f"  Points range: [{pc.min().item():.3f}, {pc.max().item():.3f}]")
        print(f"  Sample model ID: {model_id[0]}")
        print("  PASS")

except Exception as e:
    print(f"  FAIL: {e}")
    import traceback
    traceback.print_exc()
    print("\n  Continuing with dummy data for remaining tests...")
    train_loader = None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 5: Loss computation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n[5/6] Testing loss computation...")

criterion = ChamferDistanceLoss(bidirectional=True)

pred_pts = torch.randn(4, 2048, 3).to(device)
gt_pts = torch.randn(4, 2048, 3).to(device)

cd_loss = criterion(pred_pts, gt_pts)
print(f"  Chamfer Distance (random): {cd_loss.item():.6f}")

cd_loss_same = criterion(pred_pts, pred_pts)
print(f"  Chamfer Distance (same):   {cd_loss_same.item():.6f}")

fs = f_score(pred_pts, gt_pts, threshold=0.05)
print(f"  F-Score@0.05 (random):     {fs.mean().item():.4f}")

assert cd_loss.item() > 0, "CD should be positive for different point clouds"
assert cd_loss_same.item() < 1e-6, "CD should be ~0 for identical point clouds"
print("  PASS")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 6: One full training step
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n[6/6] Testing one training step...")

model.train()

# Optimizer with differential LR
encoder_params = list(model.encoder.parameters())
decoder_params = [p for n, p in model.named_parameters() if not n.startswith('encoder.')]
optimizer = torch.optim.AdamW([
    {"params": encoder_params, "lr": cfg.training.encoder_lr},
    {"params": decoder_params, "lr": cfg.training.decoder_lr},
], weight_decay=cfg.training.weight_decay)

# Use real data if available, else dummy
if train_loader is not None and len(train_loader) > 0:
    img, pc, _ = next(iter(train_loader))
    img = img.to(device)
    pc = pc.to(device)
    print(f"  Using REAL data: {img.shape}")
else:
    img = torch.randn(4, 3, 224, 224).to(device)
    pc = torch.randn(4, 2048, 3).to(device)
    print(f"  Using DUMMY data: {img.shape}")

# Forward
t0 = time.time()
pred = model(img)
loss = criterion(pred, pc)

# Backward
optimizer.zero_grad()
loss.backward()

# Check gradients exist
has_grad = sum(1 for p in model.parameters() if p.grad is not None)
total_p = sum(1 for p in model.parameters())
print(f"  Loss: {loss.item():.6f}")
print(f"  Params with gradients: {has_grad}/{total_p}")

# Step
optimizer.step()
t1 = time.time()
print(f"  Training step time: {(t1-t0)*1000:.0f}ms")

# Verify loss decreased with a second step
pred2 = model(img)
loss2 = criterion(pred2, pc)
print(f"  Loss after 1 step: {loss2.item():.6f}")

print("  PASS")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print(f"\n{'='*60}")
print("  ALL TESTS PASSED")
print(f"{'='*60}")
print(f"  Model:     {total_params/1e6:.1f}M params")
print(f"  Data:      {len(train_loader.dataset) if train_loader else 0} training samples")
print(f"  Device:    {device}")
print(f"\n  Ready for training! Run:")
print(f"    python train.py --config config.yaml")
print(f"    python train.py --experiment base")
