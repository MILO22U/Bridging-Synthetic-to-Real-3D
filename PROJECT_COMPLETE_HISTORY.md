# Complete Project History — Day 0 to Present
## Bridging the Synthetic-to-Real Gap in 3D Object Reconstruction
### AI 535 — Deep Learning | Mrinal Bharadwaj | March 2026

---

## Timeline Overview

| Date | Phase | Key Event |
|------|-------|-----------|
| Early March | Phase 1 | Data download & preprocessing |
| Early March | Phase 2 | First training (depth GT — bad) |
| Mid March | Phase 3 | Cap3D PLY ground truth (good) |
| ~March 14-15 | Phase 4 | Training with Cap3D GT (best results) |
| ~March 15-16 | Phase 5 | Domain adaptation experiments |
| March 16 | Phase 6 | ResNet-34 + 200 epoch attempt |
| March 17 | Phase 7 | Bug discovery & fix session |
| March 17-18 | Phase 8 | 2048-pt retraining (40 epochs) |
| March 19-22 | Phase 9 | Resume training to 100 epochs |
| March 23 | Phase 10 | All experiments rerun with fixed code |

---

## Phase 1: Initial Setup & Data Preparation

### Data Sources
1. **ShapeNet R2N2 Renderings** — 13 object categories, ~45K objects, 24 views each
   - Categories: airplane, bench, cabinet, car, chair, display, lamp, loudspeaker, rifle, sofa, table, telephone, watercraft
   - Synset IDs: 02691156, 02828884, 02933112, 02958343, 03001627, 03211117, 03636649, 03691459, 04090263, 04256520, 04379243, 04401088, 04530566

2. **Cap3D Rendered Images** — 8-view renders with depth maps from HuggingFace

3. **Cap3D PLY Point Clouds** — 52,472 objects, 16384 points each (mesh-sampled)

### Data Processing Pipeline
1. `download_shapenet_renders.py` — Downloaded Cap3D rendered images
2. `restructure_renders.py` — Organized into `renders/<synset>/<model_id>/image_XXXX.png`
3. `preresize_images.py` — Pre-resized 512→224 pixels for training speed → `renders_224/`
4. `download_cap3d.py` — Downloaded Cap3D PLY point clouds
5. `convert_cap3d_ply_v2.py` — Converted ASCII PLY → NPY format (2048×3 each)
6. `check_pointclouds.py` — Verified GT quality visually

### Data Layout
```
D:\DL\data\
├── shapenet\
│   ├── renders\           # 55 categories, symlinked from images
│   ├── renders_224\       # Pre-resized to 224×224 JPG
│   ├── ShapeNetRendering\ # R2N2 original 13 categories
│   └── images\            # Cap3D raw extracted images
├── cap3d\
│   └── point_clouds\      # ~39K .npy files (2048×3 each)
├── real_photos\           # 27 phone photos for qualitative eval
├── real_domain_images\    # 500 real images for DANN training
└── backgrounds\           # For augmentation (empty)
```

### Initial Cap3D Download Issue
- `Cap3D_pcs.npz` download initially failed — file was only 15 bytes
- Switched to downloading individual PLY files instead
- Generated point clouds from depth maps first (`generate_pointclouds.py`) — **these were BAD**
  - Multi-view back-projection had camera extrinsic alignment issues
  - Resulted in scattered, disconnected clusters instead of solid shapes

---

## Phase 2: First Training Runs (Depth-Projected GT — BAD)

The depth-projected ground truth was **fundamentally broken** — scattered disconnected clusters instead of solid shapes.

### Training Run 1: 5 categories, depth GT
- Early stopping at epoch 37
- **CD: 0.1767 | F@0.05: 0.0221**
- Terrible — GT was garbage

### Training Run 2: 5 categories, depth GT, lower LR
- Also early stopped
- **CD: 0.0154 | F@0.05: 0.4383**
- Better but still limited by bad GT

---

## Phase 3: Fixed Ground Truth with Cap3D PLY Files

### The Fix
- Successfully downloaded Cap3D PLY point clouds (52,472 objects, 16384 pts each)
- Converted PLY → NPY using `convert_cap3d_ply_v2.py` (handled ASCII PLY format)
- These are **proper mesh-sampled point clouds** — clean, solid 3D shapes
- Verified quality with `check_pointclouds.py`
- Sub-sampled from 16384 → 2048 points per object

### Impact
This was the single biggest improvement — switching from depth-projected to mesh-sampled GT improved CD by **2.6×** with no model changes.

---

## Phase 4: Retrained with Cap3D GT (GREAT RESULTS)

### Training Run 3: 5 categories, Cap3D GT, ResNet-18, 100 epochs
- Best epoch: 85
- **CD: 0.0059 | F@0.05: 0.6807**
- 2.6× better CD than depth GT version
- Checkpoint: `checkpoints/cap3d_resnet18_best.pt`

### Training Run 4: 13 categories, Cap3D GT, ResNet-18, 100 epochs
- 31,832 training samples
- **CD: 0.0161 | F@0.05: 0.4235**
- Slightly worse (expected — harder problem with more categories)
- Checkpoint: `checkpoints/13cat_best.pt`

### Pix2Vox Baseline
- Implemented `pix2vox_baseline.py` — 3D CNN approach
- **CD: 0.0911 | F@0.05: 0.1857**
- Our transformer model is **6× better**

---

## Phase 5: Domain Adaptation Experiments (OLD Code — Had Bugs)

All run with OLD training code that had the flip bug, single LR, etc.

### Strategy 1: Training-Time Augmentation
- Heavy color jitter, blur, random erasing during training
- Config: `config_augmented.yaml`
- **CD: 0.0155 | F@0.05: 0.4333** (on old depth GT)
- Checkpoint: `checkpoints/augmented_best.pt`

### Strategy 2: Test-Time Augmentation (TTA)
- 10 augmented views averaged at inference
- Used `TestTimeAugmentation` class in `strategies.py`
- **CD: 0.0058 | F@0.05: 0.6822** (on 5-cat Cap3D)
- Best overall result (free improvement, no retraining)

### Strategy 3: DANN (Domain Adversarial Neural Network)
- `train_dann.py` — fine-tuned with real images for domain-invariant features
- `DomainDiscriminator` in `strategies.py`
- **CD: 0.0157 | F@0.05: 0.4344** (on old depth GT)
- Checkpoint: `checkpoints/dann/best_model.pt`

### Strategy 4: AdaIN Style Transfer
- `eval_adain.py` — VGG-based feature statistics matching
- `run_adain.py` — Simple pixel-level channel statistics matching
- **CD: 0.0329** — Hurts synthetic performance (expected, designed for real only)
- Visualizations in `visualizations/adain_experiment/`

### Results Summary (Phase 5)

| Method | CD ↓ | F@0.05 ↑ | Params | Notes |
|--------|------|----------|--------|-------|
| Pix2Vox baseline | 0.0911 | 0.1857 | 16.6M | 3D CNN |
| Ours (5-cat depth GT) | 0.0154 | 0.4383 | 17.0M | Bad GT |
| **Ours (5-cat Cap3D)** | **0.0059** | **0.6807** | 17.0M | Best old code |
| Ours + TTA | 0.0058 | 0.6822 | 17.0M | Marginal boost |
| Ours + Augmentation | 0.0155 | 0.4333 | 17.0M | Old depth GT |
| Ours + DANN | 0.0157 | 0.4344 | 17.0M | Old depth GT |
| Ours + AdaIN | 0.0329 | — | 17.0M | Hurts synthetic |
| Ours (13-cat) | 0.0161 | 0.4235 | 17.0M | Old code |

---

## Phase 6: ResNet-34 + 200 Epochs Attempt

- Started training with `config_improved.yaml`
- ResNet-34 encoder (larger), 200 epochs, 13 categories
- 29.2M parameters
- **This run was interrupted by the bug-fix session**
- Partial checkpoint in `checkpoints/improved/`

---

## Phase 7: Bug Discovery & Fix Session (March 17, 2026)

### The Trigger
User uploaded real photo inference results showing model producing **diffuse blob** predictions on real images. Investigation revealed **multiple critical bugs** in the training code.

### Bug 1 (CRITICAL): RandomHorizontalFlip
- **File:** `dataset.py`, line 77
- **Problem:** `RandomHorizontalFlip` flipped images but NOT GT point clouds
- 50% of training data had misaligned supervision
- Model learned to predict symmetric blobs as a compromise
- **Fix:** Removed `RandomHorizontalFlip` from `get_train_transform()`

### Bug 2 (CRITICAL): No Differential Learning Rate
- **File:** `train.py`, lines 256-259
- **Problem:** Same LR (1e-4) for pretrained encoder AND random decoder
- Encoder's ImageNet features got destroyed in first few epochs
- **Fix:** Parameter groups with different LRs:
  ```python
  optimizer = AdamW([
      {'params': encoder_params, 'lr': base_lr * 0.2},   # 2e-5
      {'params': decoder_params, 'lr': base_lr * 5.0},   # 5e-4
  ])
  ```

### Bug 3: self_attn_layers = 6 (too many)
- **File:** `config.yaml`, line 18
- Guide recommended 4, config had 6
- 2048×2048 attention matrices × 6 layers = optimization difficulty
- **Fix:** Changed to 4 self-attention layers

### Bug 4: Warmup/Scheduler Conflict
- **File:** `train.py`, lines 303-306
- Manual warmup conflicted with CosineAnnealingLR internal counter
- LR dropped prematurely at epoch 5
- **Fix:** `SequentialLR(LinearLR + CosineAnnealingLR)`

### Bug 5: evaluate_reconstruction() Wrong API
- **File:** `losses.py`
- Called `chamfer_distance(pred, gt, reduce='mean')` — should be `bidirectional=True`
- Called `f_score()` expecting tuple `(fs, prec, rec)` — returns single tensor
- Validation crashed silently → best_model.pt was stuck at epoch 0
- **Fix:** Correct API calls + `.mean().item()` for f_score

### Bug 6: ChamferLoss() Constructor
- Called `ChamferLoss(reduce='mean')` — no such parameter
- **Fix:** `ChamferLoss()` with no args

### Bug 7: `total_mem` Typo
- Should be `total_memory`
- **Fix:** Corrected attribute name

### Additional Fix: TTA RandomHorizontalFlip
- `strategies.py` `TestTimeAugmentation` class also had flip without flipping predictions
- **Fix:** Removed `RandomHorizontalFlip` from TTA augmentation pipeline

### Critical Discovery: best_model.pt was Epoch 0
- `checkpoints/base_pretrained/best_model.pt` contained epoch 0 with val_cd=0.024
- The actual best was in `epoch_100.pt` with val_cd=0.0059
- Root cause: Bug 5 — validation crashed, so best-model-saving never triggered
- **Fix:** Copied `epoch_100.pt` → `best_model.pt`

---

## Phase 8: 2048-Point Retraining (March 17-18)

### Setup
- Created `config_2048.yaml` — 2048 output points (doubled from 1024)
- VRAM test showed max batch_size=12 for 2048 pts on 16GB card
- Created `launch_train.py` for log file output
- Added tqdm progress bars + ETA to `train.py`

### Training Configuration
```yaml
model:
  num_query_tokens: 2048
  self_attn_layers: 4
training:
  batch_size: 12
  num_epochs: 100
  learning_rate: 1.0e-4
  warmup_epochs: 2
  mixed_precision: true
data:
  13 categories, 31832 training samples
```

### Training Run (Epochs 1-40)
- Started: March 17 evening
- Performance: ~2.1 it/s, ~21 min/epoch
- **Epoch 40 results (first completion):**
  - Train Loss: 0.006356
  - Val CD: 0.009296
  - F@0.01: 0.0425 | F@0.02: 0.1987 | F@0.05: 0.6642

### Convergence History (Best Model Saves)
| Epoch | Val CD | Improvement |
|-------|--------|-------------|
| 1 | 0.097805 | — |
| 3 | 0.023438 | -76% |
| 4 | 0.022294 | -5% |
| 6 | 0.014291 | -36% |
| 8 | 0.012914 | -10% |
| 10 | 0.012392 | -4% |
| 12 | 0.011876 | -4% |
| 17 | 0.010851 | -9% |
| 19 | 0.010607 | -2% |
| 21 | 0.010461 | -1% |
| 27 | 0.009838 | -6% |
| 31 | 0.009482 | -4% |
| 36 | 0.009313 | -2% |
| 40 | 0.009296 | -0.2% |

---

## Phase 9: Resume Training to 100 Epochs (March 19-22)

### Modifications for Resume
- Fixed scheduler resume: fast-forward cosine scheduler on checkpoint load
- Added overall epoch progress bar
- Added file logging for PowerShell tailing (`Get-Content -Tail -Wait`)
- Encoder params: 11,176,512 | Decoder params: 6,328,323

### Training Continuation
- Resumed from epoch 40 checkpoint
- Trained epochs 41-100 (~60 additional epochs)
- Completed: March 22, 2026 at 21:12

### Final Results (Epoch 100)
```json
{
  "chamfer_distance": 0.008560,
  "cd_pred_to_gt": 0.002696,
  "cd_gt_to_pred": 0.005864,
  "f_score@0.01": 0.0448,
  "f_score@0.02": 0.2068,
  "f_score@0.05": 0.6858,
  "best_val_cd": 0.008105,
  "total_params": 17504835
}
```

### Last Two Epochs
```
Epoch 99/100 (enc_lr=1.02e-06, dec_lr=1.51e-06)
  Train Loss: 0.005439 | Val CD: 0.008418 | F@0.05: 0.6865

Epoch 100/100 (enc_lr=1.00e-06, dec_lr=1.13e-06)
  Train Loss: 0.005414 | Val CD: 0.008402 | F@0.05: 0.6855

Best Val CD: 0.008105 (saved at ~epoch 98)
```

---

## Phase 10: All Experiments Rerun (March 23)

### Evaluations Completed
| Model | CD ↓ | F@0.01 ↑ | F@0.02 ↑ | F@0.05 ↑ |
|-------|------|----------|----------|----------|
| Old 1024-pt (100ep, TTA) | 0.00582 | 0.0223 | 0.1428 | 0.6822 |
| New 2048-pt (100ep, no TTA) | 0.00858 | 0.0445 | 0.2053 | 0.6829 |
| New 2048-pt (100ep, TTA) | 0.00917 | — | — | 0.6761 |

**Key finding:** TTA slightly *hurts* the 2048-pt model (CD 0.00917 vs 0.00858). The larger model is already well-calibrated.

### Experiments In Progress (March 23)
Running sequentially via `run_all_experiments.py`:
1. AdaIN alpha=0.3 → `visualizations2/adain_alpha0.3/`
2. AdaIN alpha=0.5 → `visualizations2/adain_alpha0.5/`
3. AdaIN alpha=0.8 → `visualizations2/adain_alpha0.8/`
4. TTA on real photos → `visualizations2/tta_real/`
5. AdaIN VGG eval → `visualizations2/adain_vgg/`
6. DANN training (20 epochs) → `checkpoints/dann_2048/`

### OOM Fixes Applied
- `run_adain.py`: Welford online stats (no memory accumulation), inference 1 image at a time
- `train_dann.py`: batch_size=8, config_2048.yaml, correct checkpoint
- `eval_adain.py`: batch_size=1 for synthetic eval, cache clearing per sample
- All scripts: `torch.cuda.empty_cache()` between samples

---

## Architecture Details

### Model: Hybrid CNN-Transformer "EASU" (17.5M parameters)

```
Input Image (224×224×3)
    │
    ▼
┌─────────────────────┐
│ ResNet-18 Encoder    │  ← ImageNet pretrained
│ (11.2M params)       │  ← LR: 2e-5 (gentle fine-tuning)
│ Output: 49×512       │
└─────────┬───────────┘
          │ Linear projection (512→256)
          ▼
┌─────────────────────┐
│ Cross-Attention      │  ← 2 layers, 8 heads
│ Bridge               │  ← 2048 learnable query tokens
│ (49 image tokens     │     attend to image features
│  → 2048 queries)     │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ Transformer Decoder  │  ← 4 layers, 8 heads
│ (Self-Attention)     │  ← LR: 5e-4 (fast learning)
│ 2048×256             │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ MLP Head             │
│ 256 → 512 → 3       │
│ Output: 2048×3       │  ← (x, y, z) point cloud
└─────────────────────┘
```

### Parameter Breakdown
- ResNet-18 Encoder: 11,176,512 (63.8%)
- Cross-Attention Bridge: ~2.1M
- Transformer Decoder: ~2.1M
- MLP Head: ~2.1M
- Decoder total: 6,328,323 (36.2%)
- **Total: 17,504,835**

### Training Configuration
- **Optimizer:** AdamW with differential LR
  - Encoder: base_lr × 0.2 = 2e-5
  - Decoder: base_lr × 5.0 = 5e-4
- **Scheduler:** SequentialLR
  - Phase 1: LinearLR warmup (2 epochs, start_factor=0.1)
  - Phase 2: CosineAnnealingLR (98 epochs, eta_min=1e-6)
- **Loss:** Bidirectional Chamfer Distance (squared L2)
- **Mixed Precision:** FP16 via GradScaler
- **Gradient Clipping:** max_norm=1.0

---

## All Checkpoints

| Checkpoint | Epoch | Points | Categories | Val CD | Size |
|-----------|-------|--------|------------|--------|------|
| `retrain_2048/best.pt` | ~98 | 2048 | 13 | **0.008105** | 201M |
| `base_pretrained/best_model.pt` | 100 | 1024 | 13 | 0.0059 | 195M |
| `cap3d_resnet18_best.pt` | 85 | 1024 | 5 | 0.0059 | 195M |
| `13cat_best.pt` | — | 1024 | 13 | 0.0161 | 195M |
| `dann/best_model.pt` | — | 1024 | 13 | 0.0157 | 195M |
| `augmented_best.pt` | — | 1024 | — | — | 195M |

---

## Lessons Learned

### 1. Spatial Augmentations Must Be Consistent
Any geometric augmentation (flip, rotation) on input MUST have corresponding transform on target. `RandomHorizontalFlip` without flipping point cloud x-coordinates = 50% contradictory supervision.

### 2. Pretrained Encoders Need Lower LR
Same LR for pretrained encoder and random decoder destroys ImageNet features. Use differential: encoder 0.2× base, decoder 5× base.

### 3. Verify Loss/Metric API Signatures
`chamfer_distance(pred, gt, reduce='mean')` silently fails — should be `bidirectional=True`. Always test with dummy data before training.

### 4. Always Inspect Checkpoints
`best_model.pt` was epoch 0 — not the actual best. Always verify: `print(ckpt['epoch'], ckpt['val_cd'])`.

### 5. Test GPU Memory Before Long Runs
batch_size=16 with 2048 points = 21GB needed on 16GB card. Run `vram_test.py` first.

### 6. Background Removal is Critical
Trained on synthetic renders with black backgrounds. Real photos need `rembg` background removal at test time — single biggest improvement for real-world inference.

### 7. Ground Truth Quality > Model Complexity
Switching from depth-projected to Cap3D mesh-sampled GT: CD 0.0154 → 0.0059 (2.6× improvement). No model change needed.

---

## File Reference

### Core Files (FIXED)
| File | Lines | Purpose |
|------|-------|---------|
| `model.py` | 433 | HybridReconstructor architecture |
| `dataset.py` | 420 | ShapeNetCap3D dataset, transforms (no flip) |
| `train.py` | 455 | Training loop (differential LR, SequentialLR) |
| `losses.py` | 151 | Chamfer Distance, F-Score, evaluation |
| `config.yaml` | 99 | Main config (4 self-attn layers) |
| `config_2048.yaml` | 101 | 2048-point retraining config |

### Evaluation & Strategies
| File | Purpose |
|------|---------|
| `evaluate.py` | Full evaluation pipeline |
| `strategies.py` | TTA, DANN, AdaIN implementations |
| `run_tta_real.py` | TTA inference on real photos |
| `run_adain.py` | Simple AdaIN experiment |
| `eval_adain.py` | VGG AdaIN evaluation |
| `train_dann.py` | DANN fine-tuning |
| `run_real_inference.py` | Real photo inference + bg removal |

### Visualization & Analysis
| File | Purpose |
|------|---------|
| `gen_synth_viz.py` | Synthetic test visualizations |
| `gen_comparison_viz.py` | Old vs new comparison |
| `plot_training_curves.py` | Training curve plots |
| `run_all_experiments.py` | Master experiment runner |

### Data Preparation (Already Executed)
| File | What it did |
|------|------------|
| `download_cap3d.py` | Downloaded Cap3D PLY files |
| `download_shapenet_renders.py` | Downloaded rendered images |
| `convert_cap3d_ply_v2.py` | PLY → NPY conversion |
| `restructure_renders.py` | Folder reorganization |
| `preresize_images.py` | 512→224 pre-resize |

---

## Visualization Outputs

### visualizations/ (Original)
- `presentation/` — 19 presentation-quality images
- `synthetic_test/` — 20 synthetic test samples (1024-pt)
- `synthetic_test_2048/` — 2048-pt model samples
- `tta_real/` — 27 TTA real photo results
- `real_inference/` — Earlier real inference
- `adain_experiment/` — AdaIN comparisons
- `comparison_old_vs_new/` — Side-by-side
- `plots/` — Training curves

### visualizations2/ (New Experiments — March 23)
- `synthetic_test/` — 30 new synthetic samples
- `real_inference/` — 27 real photos (new model)
- `comparison/` — Updated comparison charts
- `tta_real/` — TTA with new model
- `adain_alpha0.3/`, `adain_alpha0.5/`, `adain_alpha0.8/` — AdaIN sweep
- `adain_vgg/` — VGG-level AdaIN evaluation
- `eval_results/` — Metrics JSON files
