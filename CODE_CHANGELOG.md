# Code Change Log — 3D Reconstruction Project
### Every code modification that led to the final results
### AI 535 | Mrinal Bharadwaj | March 2026

---

## Change 0 — Data Pipeline Scripts (Day 0)
### Files Created

**Purpose:** Download, convert, and organize training data.

| Script | What It Does |
|--------|-------------|
| `download_shapenet_renders.py` | Download Cap3D rendered images from HuggingFace |
| `restructure_renders.py` | Organize into `renders/<synset>/<model_id>/image_XXXX.png` |
| `preresize_images.py` | Resize 512→224px → `renders_224/` for training speed |
| `download_cap3d.py` | Download Cap3D PLY point clouds (52,472 objects) |
| `convert_cap3d_ply_v2.py` | Convert ASCII PLY → NPY format (2048×3 per object) |
| `check_pointclouds.py` | Visual verification of GT quality |
| `generate_pointclouds.py` | Depth-map back-projection (ABANDONED — produced bad GT) |

**Key decision:** Depth-projected GT was fundamentally broken (scattered clusters). Switching to Cap3D mesh-sampled PLY files was the single biggest improvement — 2.6× CD improvement with zero model changes.

---

## Change 1 — Bug Fix: Remove RandomHorizontalFlip (Day 6)
### File: `dataset.py`, line ~77

**Before (BUGGY):**
```python
def get_train_transform(cfg_aug, use_random_bg=True):
    transforms_list = []
    transforms_list.extend([
        T.Resize((224, 224)),
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),  # <-- BUG: flips image but NOT GT point cloud
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        # ...
    ])
```

**After (FIXED):**
```python
def get_train_transform(cfg_aug, use_random_bg=True):
    """
    FIX: Removed RandomHorizontalFlip — it was flipping images without
    flipping the corresponding GT point clouds, creating contradictory
    supervision signals that caused the model to predict symmetric blobs.
    """
    transforms_list = []
    transforms_list.extend([
        T.Resize((224, 224)),
        T.RandomResizedCrop(224, scale=cfg_aug.get('random_crop_scale', (0.8, 1.0))),
        T.ColorJitter(
            brightness=cfg_aug.get('color_jitter', 0.4),
            contrast=cfg_aug.get('color_jitter', 0.4),
            saturation=cfg_aug.get('color_jitter', 0.4),
            hue=min(cfg_aug.get('color_jitter', 0.4) / 2, 0.5)
        ),
        # NOTE: RandomHorizontalFlip REMOVED — cannot flip images without also
        # flipping GT point cloud x-coordinates. This was a critical bug.
    ])
```

**Impact:** This was the #1 bug. 50% of training data had misaligned supervision — the model learned to predict symmetric blobs as a compromise between flipped and non-flipped targets.

---

## Change 2 — Bug Fix: Differential Learning Rate (Day 6)
### File: `train.py`, lines 254–275

**Before (BUGGY):**
```python
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
```

**After (FIXED):**
```python
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
```

**Impact:** Using the same LR (1e-4) for both pretrained ResNet encoder and random decoder destroyed ImageNet features in the first few epochs. With differential LR, encoder features are preserved while decoder learns quickly.

---

## Change 3 — Bug Fix: Self-Attention Layers 6→4 (Day 6)
### File: `config.yaml`, line 20

**Before:**
```yaml
self_attn_layers: 6    # Too many — 2048×2048 attention matrices × 6
```

**After:**
```yaml
self_attn_layers: 4    # FIX: reduced for better optimization
```

**Impact:** 6 self-attention layers with 2048 query tokens created optimization difficulty. Reduced to 4 per project guide recommendations.

---

## Change 4 — Bug Fix: Warmup/Scheduler Conflict (Day 6)
### File: `train.py`, lines 283–299

**Before (BUGGY):**
```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Manual warmup in training loop:
for epoch in range(num_epochs):
    if epoch < warmup_epochs:
        for pg in optimizer.param_groups:
            pg['lr'] = base_lr * (epoch + 1) / warmup_epochs
    scheduler.step()  # <-- BUG: scheduler counter advances during warmup too
```

**After (FIXED):**
```python
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
```

**Impact:** Manual warmup conflicted with CosineAnnealingLR's internal counter, causing LR to drop prematurely at epoch 5. `SequentialLR` properly chains warmup and cosine phases.

---

## Change 5 — Bug Fix: evaluate_reconstruction() Wrong API (Day 6)
### File: `losses.py`

**Before (BUGGY):**
```python
def evaluate_reconstruction(pred, gt, thresholds=[0.01, 0.02, 0.05]):
    with torch.no_grad():
        cd_loss = chamfer_distance(pred, gt, reduce='mean')  # <-- BUG: 'reduce' not a valid param
        results = {'chamfer_distance': cd_loss}
        for t in thresholds:
            fs, prec, rec = f_score(pred, gt, threshold=t)   # <-- BUG: f_score returns tensor, not tuple
            results[f'f_score@{t}'] = fs
        return results
```

**After (FIXED):**
```python
def evaluate_reconstruction(pred, gt, thresholds=[0.01, 0.02, 0.05]):
    with torch.no_grad():
        cd_loss, cd_p2g, cd_g2p = chamfer_distance(pred, gt, bidirectional=True)
        results = {
            'chamfer_distance': cd_loss.item(),
            'cd_pred_to_gt': cd_p2g.item(),
            'cd_gt_to_pred': cd_g2p.item(),
        }
        for t in thresholds:
            fs = f_score(pred, gt, threshold=t)
            results[f'f_score@{t}'] = fs.mean().item()
        return results
```

**Impact:** The old code crashed validation silently. Since validation never completed, the best-model checkpoint was stuck at epoch 0 (val_cd=0.024) instead of the actual best (val_cd=0.0059). This bug made all saved "best" checkpoints useless.

---

## Change 6 — Bug Fix: ChamferLoss Constructor (Day 6)
### File: `losses.py`

**Before:**
```python
criterion = ChamferDistanceLoss(reduce='mean')  # <-- 'reduce' not a valid param
```

**After:**
```python
criterion = ChamferDistanceLoss()  # bidirectional=True is default
```

---

## Change 7 — Bug Fix: `total_mem` Typo (Day 6)
### File: `train.py`

**Before:**
```python
total_mem = torch.cuda.get_device_properties(0).total_mem  # <-- AttributeError
```

**After:**
```python
total_mem = torch.cuda.get_device_properties(0).total_memory
```

---

## Change 8 — Bug Fix: TTA RandomHorizontalFlip (Day 6)
### File: `strategies.py`, TestTimeAugmentation class

**Before (BUGGY):**
```python
self.augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # <-- BUG: flips without flipping predictions
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
])
```

**After (FIXED):**
```python
self.augment = transforms.Compose([
    # NOTE: RandomHorizontalFlip REMOVED — flipping image without flipping
    # the predicted point cloud causes misalignment
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
])
```

---

## Change 9 — New Config: 2048-Point Training (Day 7)
### File: `config_2048.yaml` (NEW)

**Key differences from `config.yaml`:**
```yaml
model:
  num_query_tokens: 2048    # was 1024 — doubled output points
  self_attn_layers: 4       # fixed (was 6 in original)

training:
  batch_size: 12            # was 16 — reduced for VRAM (2048 pts needs more memory)
  warmup_epochs: 2          # was 5 — shorter warmup
  mixed_precision: true     # FP16 for VRAM savings

data:
  num_points: 2048          # match model output
  train_categories: [all 13 synsets]

logging:
  save_dir: "./checkpoints/retrain_2048"
```

**Impact:** VRAM test showed batch_size=12 is maximum for 2048 pts on 16GB RTX 4070 Ti Super. batch_size=16 would need 21GB.

---

## Change 10 — Training Resume: Scheduler Fast-Forward (Day 8)
### File: `train.py`

**Added code for resuming from checkpoint:**
```python
# After loading checkpoint, fast-forward scheduler to correct epoch
if start_epoch > 0:
    for _ in range(start_epoch):
        scheduler.step()
    print(f"  Fast-forwarded scheduler to epoch {start_epoch}")
```

**Also added:**
```python
# File logging for PowerShell monitoring
import logging
file_handler = logging.FileHandler('training_log.txt')
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
logger.addHandler(file_handler)
# User monitors with: Get-Content training_log.txt -Tail 30 -Wait
```

---

## Change 11 — OOM Fix: run_adain.py Reordering (Day 9)
### File: `run_adain.py`

**Before (OOM at sample 10/27):**
```python
def main():
    # Load model on GPU
    model = HybridReconstructor(cfg['model']).to(device)

    # Create dataloaders (num_workers=4 forks process memory)
    train_loader, _, _ = create_dataloaders(cfg)

    # Compute stats (dataloader + model both on GPU = OOM)
    syn_mean, syn_std = compute_stats(train_loader)

    # Load photos and run inference (still has dataloader in memory)
    for img in photos:
        pred = model(img)  # <-- OOM here
```

**After (FIXED — sequential CPU→GPU):**
```python
def main():
    # Step 1: Compute stats on CPU FIRST (no model on GPU yet)
    stats_cfg['training']['batch_size'] = 4
    stats_cfg['training']['num_workers'] = 0  # no worker forks
    train_loader, _, _ = create_dataloaders(stats_cfg)
    syn_mean, syn_std = compute_synthetic_pixel_stats(train_loader, max_batches=50)
    del train_loader, stats_cfg  # FREE ALL MEMORY
    gc.collect()
    torch.cuda.empty_cache()

    # Step 2: Load and style-transfer photos on CPU
    styled_images = simple_adain(real_images, syn_mean, syn_std, alpha=args.alpha)
    del syn_mean, syn_std
    gc.collect()

    # Step 3: NOW load model on GPU (nothing else occupying VRAM)
    model = HybridReconstructor(cfg['model']).to(device)

    # Step 4: Inference ONE image at a time
    for i in range(len(names)):
        img = real_images[i:i+1].to(device)
        pred = model(img).cpu()
        del img
        torch.cuda.empty_cache()
```

**Also changed:** Used Welford's online algorithm for stats (no memory accumulation):
```python
def compute_synthetic_pixel_stats(dataloader, max_batches=100):
    """Welford's online algorithm — O(1) memory regardless of dataset size."""
    n = 0
    mean_acc = torch.zeros(1, 3, 1, 1)
    m2_acc = torch.zeros(1, 3, 1, 1)

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        images = batch[0]
        batch_mean = images.mean(dim=[0, 2, 3], keepdim=True)
        batch_var = images.var(dim=[0, 2, 3], keepdim=True)
        batch_n = images.shape[0]

        delta = batch_mean - mean_acc
        total = n + batch_n
        mean_acc = mean_acc + delta * batch_n / total
        m2_acc = m2_acc + batch_var * batch_n + delta ** 2 * n * batch_n / total
        n += batch_n

    return mean_acc, torch.sqrt(m2_acc / n) + 1e-8
```

**Impact:** Root cause was `create_dataloaders()` with `num_workers=4` forking process memory while model occupied GPU. Fix: compute stats with 0 workers on CPU, delete everything, then load model.

---

## Change 12 — eval_adain.py: Batch-Size-1 Evaluation (Day 9)
### File: `eval_adain.py`, line 133

```python
# Process test set one sample at a time to fit in 16GB VRAM
# (model + VGG both on GPU simultaneously)
loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

for img, pc, _ in tqdm(loader, desc="AdaIN eval"):
    # Original prediction
    pred_orig = model(img.to(device))

    # Style-transferred prediction (VGG + model on same GPU)
    styled = adain_model.transfer_to_synthetic_style(img.to(device), syn_stats)
    pred_adain = model(styled)

    # Compute CD for both
    cd_o, _, _ = chamfer_distance(pred_orig, pc.to(device))
    cd_a, _, _ = chamfer_distance(pred_adain, pc.to(device))

    del img, pc, pred_orig, pred_adain, styled
    torch.cuda.empty_cache()  # <-- Added: clear cache every sample
```

---

## Change 13 — run_dann_v2.py: DANN with 2048 Points (Day 9)
### File: `run_dann_v2.py` (NEW)

**Key settings for 16GB VRAM:**
```python
# Reduced batch size for 2048 points + discriminator on GPU
cfg['training']['batch_size'] = 8      # was 12 for training, 8 for DANN (extra discriminator memory)
cfg['training']['num_epochs'] = 20     # fine-tuning, not full retraining
cfg['domain_adaptation']['dann_enabled'] = True

# Load pretrained 2048-pt model as starting point
checkpoint = 'checkpoints/retrain_2048/best.pt'
```

---

---

### Day 10 — AdaIN OOM Fix & Final Experiments (March 23, 2026)

#### Change 14 — Fix run_adain.py CUDA OOM (CRITICAL)

**File:** `run_adain.py` · **Impact:** All 3 AdaIN alpha experiments now complete successfully

The inference loop accumulated CUDA memory across iterations. After ~10 images, fragmentation caused OOM on 16GB VRAM. Fix: delete intermediate tensors immediately after `.cpu()`, add `gc.collect()` per iteration, and delete checkpoint dict after loading weights.

```diff
  for i in range(len(names)):
      with torch.no_grad():
          img_orig = real_images[i:i+1].to(device)
          pred_orig = model(img_orig).cpu()
          all_pred_original.append(pred_orig)
-         del img_orig
+         del img_orig, pred_orig

          img_styled = styled_images[i:i+1].to(device)
          pred_styled = model(img_styled).cpu()
          all_pred_styled.append(pred_styled)
-         del img_styled
+         del img_styled, pred_styled

-     torch.cuda.empty_cache()
+     gc.collect()
+     torch.cuda.empty_cache()
```

Also added after model loading:
```diff
  model.load_state_dict(ckpt['model_state_dict'])
  model.eval()
  print(f"  Loaded epoch {ckpt['epoch']}")
+ del ckpt
+ gc.collect()
+ torch.cuda.empty_cache()
```

#### Change 15 — Updated comparison charts with all experiment results

**File:** `generate_viz2.py` · **Impact:** Charts now include all strategies (baseline, TTA, DANN partial, AdaIN VGG, AdaIN pixel)

Previously only showed old 1024-pt vs new 2048-pt. Now includes full domain adaptation strategy comparison with CD bar chart, F-score grouped bars, and summary table with notes column.

#### Change 16 — Updated README with final experiment results

**File:** `README.md` · **Impact:** Accurate reporting of all experiment outcomes

- AdaIN pixel experiments: FAIL → PASS (all 3 alphas, 27 comparisons each)
- DANN training: RUNNING → PARTIAL (48% of epoch 1 after 14 hrs, infeasible)
- Updated strategy comparison table with actual numbers
- Added `dann_2048/best_model.pt` to checkpoints table

---

## Summary: Cumulative Impact of Changes

| Change | Type | CD Impact |
|--------|------|-----------|
| Cap3D PLY GT (Change 0) | Data | 0.0154 → 0.0059 (2.6× better) |
| Remove flip bug (Change 1) | Critical fix | Eliminated symmetric blob predictions |
| Differential LR (Change 2) | Critical fix | Preserved pretrained encoder features |
| self_attn 6→4 (Change 3) | Config fix | Easier optimization |
| SequentialLR (Change 4) | Scheduler fix | Correct LR schedule |
| evaluate API (Change 5) | Validation fix | Best checkpoints actually saved correctly |
| 2048 points (Change 9) | Architecture | F@0.01 doubled (0.0223→0.0448) |
| OOM fixes (Changes 11-12) | Infrastructure | Enabled AdaIN/VGG experiments on 16GB |
| AdaIN OOM fix (Change 14) | Infrastructure | All 3 AdaIN alpha experiments pass |

**Final best result:** CD 0.008105, F@0.01 0.0448, F@0.05 0.6858 (2048-pt, 100 epochs, all bugs fixed)

### Domain Adaptation Results Summary

| Strategy | CD ↓ | Verdict |
|----------|------|---------|
| Baseline (2048-pt) | **0.00856** | Best |
| + TTA (synthetic) | 0.00917 | Slightly worse |
| + DANN (partial) | 0.00942 | Worse (incomplete training) |
| + AdaIN VGG | 0.05052 | 6× worse (broken approach) |
| + AdaIN pixel | — | Qualitative only (real photos) |

**Conclusion:** No domain adaptation strategy improved over the baseline on synthetic test data. TTA helps on real photos (reduces prediction spread in 8/10 cases). The baseline 2048-pt model with all bug fixes remains the best.
