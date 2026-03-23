# COMPLETE PROJECT BRIEFING
## Bridging the Synthetic-to-Real Gap in 3D Object Reconstruction
### AI 535 — Deep Learning | Mrinal Bharadwaj | March 2026

---

## 1. PROJECT OVERVIEW

**Goal:** Single-image 3D point cloud reconstruction. Train on synthetic ShapeNet renders, measure domain gap on real photos, try 4 strategies to close the gap.

**Architecture:** Hybrid CNN-Transformer ("EASU")
- ResNet-18/34 encoder (ImageNet pretrained) → 49×512 tokens
- Cross-Attention Bridge (2 layers, 8 heads) → 2048×256 queries
- Transformer Decoder (self-attention) → 2048×256
- MLP Head → 2048×3 point cloud

**Hardware:** RTX 4070 Ti Super (16GB VRAM), Windows, `D:\DL\`, conda env `recon3d`

---

## 2. COMPLETE TIMELINE OF WHAT HAPPENED

### Phase 1: Initial Setup & Data Preparation
- Downloaded ShapeNet R2N2 renderings (13 categories, ~45K objects, 24 views each)
- Downloaded Cap3D rendered images from HuggingFace (separate 8-view renders with depth maps)
- Restructured renders into `renders/<synset>/<model_id>/image_XXXX.png` format
- **Cap3D_pcs.npz download initially failed** (file was only 15 bytes)
- Generated point clouds from depth maps using `generate_pointclouds.py` → multi-view back-projection
- Also tried `fix_pointclouds.py` (single-view) and `fix_pointclouds_multiview.py`

### Phase 2: First Training Runs (Depth-Projected GT — BAD)
The depth-projected ground truth was **broken** — scattered disconnected clusters instead of solid shapes. Camera extrinsics weren't aligning properly across views.

**Training Run 1: 5 categories, depth GT**
- Early stopping at epoch 37
- CD: 0.1767 | F@0.05: 0.0221
- Terrible results because GT was garbage

**Training Run 2: 5 categories, depth GT, lower LR**
- Also early stopped
- CD: 0.0154 | F@0.05: 0.4383
- Better but still limited by bad GT

### Phase 3: Fixed Ground Truth with Cap3D PLY Files
- Successfully downloaded Cap3D PLY point clouds (52,472 objects, 16384 pts each)
- Converted PLY → NPY using `convert_cap3d_ply_v2.py` (ASCII PLY format)
- These are **proper mesh-sampled point clouds** — clean, solid 3D shapes
- Verified quality with `check_pointclouds.py`

### Phase 4: Retrained with Cap3D GT (GREAT RESULTS)
**Training Run 3: 5 categories, Cap3D GT, ResNet-18, 100 epochs**
- Best epoch: 85
- **CD: 0.0059 | F@0.05: 0.6807**
- 2.6x better CD than depth GT version

**Training Run 4: 13 categories, Cap3D GT, ResNet-18, 100 epochs**
- 31,832 training samples
- CD: 0.0161 | F@0.05: 0.4235
- Slightly worse (expected — harder problem with more categories)

### Phase 5: All Domain Adaptation Experiments (OLD train.py — had bugs)

These were all run with the OLD training code that had the RandomHorizontalFlip bug, single LR, etc. Results on 5-category Cap3D data:

| Experiment | CD ↓ | F@0.05 ↑ | Notes |
|-----------|------|----------|-------|
| **Pix2Vox baseline** | 0.0911 | 0.1857 | 3D CNN, same data |
| Ours (5-cat depth GT) | 0.0154 | 0.4383 | Bad GT |
| **Ours (5-cat Cap3D)** | **0.0059** | **0.6807** | Best result |
| Ours + TTA | 0.0058 | 0.6822 | Marginal improvement |
| Ours + Augmentation | 0.0155 | 0.4333 | On old depth GT |
| Ours + DANN | 0.0157 | 0.4344 | On old depth GT |
| Ours + AdaIN (synthetic) | 0.0329 | — | Hurts synthetic (expected) |
| Ours (13-cat Cap3D) | 0.0161 | 0.4235 | More categories |

### Phase 6: Attempted ResNet-34 + 200 Epochs + 13 Categories
- Started training with `config_improved.yaml` (ResNet-34, 200 epochs, 13 cats)
- 29.2M parameters, ~8-9 hours estimated
- **This run was interrupted/overridden by the bug-fix session below**

### Phase 7: Bug Discovery & Fix Session (CURRENT — March 17, 2026)
User uploaded real photo inference results showing model producing **diffuse blob** predictions on real images. Investigation found **multiple critical bugs** in the training code:

**Bug 1 (CRITICAL): RandomHorizontalFlip**
- `dataset.py` flipped images but NOT GT point clouds
- 50% of training data had misaligned supervision
- **Fixed:** Removed from `get_train_transform()`

**Bug 2 (CRITICAL): No Differential Learning Rate**
- `train.py` used same LR for pretrained encoder and random decoder
- Encoder's ImageNet features got destroyed
- **Fixed:** Encoder LR = 2e-5, Decoder LR = 5e-4

**Bug 3: self_attn_layers = 6 (too many)**
- Guide recommended 4, config had 6
- **Fixed:** Changed to 4

**Bug 4: Warmup/Scheduler conflict**
- Manual warmup conflicted with CosineAnnealingLR
- **Fixed:** Using SequentialLR(LinearLR + CosineAnnealingLR)

**Bug 5: evaluate_reconstruction() wrong API**
- Called `chamfer_distance(pred, gt, reduce='mean')` — should be `bidirectional=True`
- Called `f_score()` expecting tuple — returns tensor
- **Fix provided but may not be applied yet — CHECK**

**Bug 6: ChamferLoss() constructor**
- Called `ChamferLoss(reduce='mean')` — should be `ChamferLoss()`
- **Fixed**

**Bug 7: `total_mem` typo**
- Should be `total_memory`
- **Fixed**

### Phase 7 Current State: Retraining in Progress
Started fresh retraining with all bug fixes applied:
```
Device: cuda
GPU: NVIDIA GeForce RTX 4070 Ti SUPER
VRAM: 17.2 GB
✓ Loaded pretrained resnet18 (ImageNet) encoder
HybridReconstructor: 17.5M parameters (was 17.0M — 4 self-attn layers vs 6 changes param count slightly)
ShapeNetCap3D [train]: 31832 samples
ShapeNetCap3D [val]: 3979 samples
ShapeNetCap3D [test]: 3979 samples

Epoch 1/100 (enc_lr=2.00e-06, dec_lr=5.00e-05)
Batch 0/1989 | Loss: 0.296665 | CD: 0.296665
```

**Config:** 13 categories, ResNet-18, 4 self-attn layers, differential LR, no flip augmentation, 100 epochs.

---

## 3. EXISTING CHECKPOINTS

| Path | Description | Quality |
|------|------------|---------|
| `checkpoints/base_pretrained/best_model.pt` | 5-cat Cap3D, epoch 85 | CD 0.0059, F@0.05 0.68 — **BEST SO FAR** |
| `checkpoints/base_pretrained/final_model.pt` | Same run, epoch 100 | Slightly worse than best |
| `checkpoints/base_pretrained/epoch_*.pt` | Periodic saves | Various |
| `checkpoints/cap3d_resnet18_best.pt` | Backup of above | Same as best_model.pt |
| `checkpoints/13cat_best.pt` | 13-cat old train.py | CD 0.0161 |
| `checkpoints/dann/best_model.pt` | DANN fine-tuned | CD 0.0157 |
| `checkpoints/improved/` | ResNet-34 attempt | May be incomplete |

**WARNING:** The `base_pretrained/` folder is being overwritten by the current retraining run. The old best model should be backed up as `cap3d_resnet18_best.pt`.

---

## 4. EXISTING OUTPUTS & VISUALIZATIONS

| Path | Contents |
|------|----------|
| `visualizations/synthetic_*.png` | Synthetic test: image + predicted + GT point clouds |
| `visualizations/real_*.png` | Real photo inference (original + bg-removed + predictions) |
| `visualizations/gt_check/` | Ground truth point cloud quality checks |
| `visualizations/adain/` | AdaIN style transfer comparisons |
| `visualizations/plots/` | Training curves, loss, LR schedule |
| `eval_results/results_base_pretrained.json` | Evaluation metrics JSON |
| `outputs/logs/base_pretrained/` | TensorBoard logs |

---

## 5. KEY FILES REFERENCE

### Used by Current Training
| File | Purpose |
|------|---------|
| `config.yaml` | YAML config (dict format) — **FIXED version** |
| `model.py` | `HybridReconstructor`, `build_model(cfg_dict)` |
| `dataset.py` | `ShapeNetCap3DDataset`, `create_dataloaders()` — **FIXED, no flip** |
| `train.py` | Training loop — **FIXED, differential LR + SequentialLR** |
| `losses.py` | `chamfer_distance()`, `ChamferDistanceLoss`, `f_score()` — **evaluate_reconstruction MAY NEED FIX** |

### Evaluation & Strategies
| File | Purpose |
|------|---------|
| `evaluate.py` | Eval script — uses **dataclass** config via `config.py` |
| `strategies.py` | TTA, DANN, AdaIN — **TTA still has RandomHorizontalFlip bug** |
| `train_dann.py` | DANN training — uses dataclass config |
| `eval_adain.py` | AdaIN evaluation |
| `run_real_inference.py` | Real photo + background removal |
| `run_adain.py` | Simple AdaIN experiment |
| `visualize.py` | Plotting utilities |
| `run_visualize.py` | Generate viz from checkpoint |
| `pix2vox_baseline.py` | Pix2Vox model + training utils |

### Unused / Legacy (DON'T TOUCH)
| File | Why |
|------|-----|
| `reconstructor.py` | Alternative model — NOT imported anywhere |
| `datasets.py` | Older dataset file — NOT imported |
| `config.py` | Dataclass config — used by evaluate.py, NOT by train.py |

### Data Preparation Scripts (ALREADY RAN — don't need again)
| File | What it did |
|------|------------|
| `download_cap3d.py` | Downloaded Cap3D PLY point clouds |
| `download_shapenet_renders.py` | Downloaded Cap3D rendered images |
| `convert_cap3d_ply_v2.py` | Converted PLY → NPY (ASCII format) |
| `restructure_renders.py` | Reorganized folder structure |
| `preresize_images.py` | Pre-resized 512→224 for speed |
| `generate_pointclouds.py` | Depth→point cloud (OLD, BAD method) |
| `fix_pointclouds.py` | Single-view depth (also bad) |
| `fix_pointclouds_multiview.py` | Multi-view depth (also bad) |

---

## 6. API SIGNATURES (CRITICAL — READ BEFORE EDITING)

```python
# losses.py
def chamfer_distance(pred, target, bidirectional=True):
    # Returns: (cd_loss, cd_p2t, cd_t2p) — all scalars
    # Uses SQUARED L2 distances

class ChamferDistanceLoss(nn.Module):
    def __init__(self, bidirectional=True):  # NO 'reduce' param
    def forward(self, pred, target) -> scalar

ChamferLoss = ChamferDistanceLoss  # alias

def f_score(pred, target, threshold=0.01):
    # Returns: (B,) tensor — NOT tuple (fs, prec, rec)
    # Uses EUCLIDEAN distances (sqrt)

# model.py
def build_model(cfg_dict):
    # Takes YAML dict, returns (HybridReconstructor, discriminator_or_None)

# dataset.py
def create_dataloaders(cfg_dict):
    # Takes YAML dict, returns (train_loader, val_loader, test_loader)
```

---

## 7. KNOWN REMAINING ISSUES

1. **`evaluate_reconstruction()` in losses.py** — May still have wrong signatures (`reduce='mean'` instead of `bidirectional=True`, tuple unpacking instead of `.mean().item()`). Training will crash at end of epoch 1 when validation runs if not fixed.

2. **TTA in strategies.py** — `TestTimeAugmentation` class still uses `RandomHorizontalFlip` which has the same flip-without-flipping-points bug. Remove it before using TTA on the new model.

3. **Background augmentation** — `config.yaml` has `use_random_backgrounds: true` but `./data/backgrounds/` may be empty. If empty, it silently does nothing (no crash, just no augmentation).

4. **No GSO data** — `D:/DL/data/gso` not found. Can't do quantitative real-world evaluation, only qualitative with phone photos.

5. **Config system confusion** — `train.py` uses YAML dict, `evaluate.py` uses dataclass. They access config differently.

6. **`autocast` deprecation warning** — Harmless. Fix: `torch.amp.autocast('cuda', ...)`.

---

## 8. WHAT TO DO NEXT

### Immediate
1. ⚠️ Verify `evaluate_reconstruction()` in losses.py is fixed (will crash validation otherwise)
2. Let current retraining finish (100 epochs, ~6-8 hours)
3. Check synthetic test metrics — expect CD 0.003-0.008 with bug fixes

### After Training
4. Run `evaluate.py` on synthetic test
5. Run `run_real_inference.py` on real photos — see if predictions improved
6. Compare with old best (CD 0.0059) — new model with fix should be similar or better

### Re-run Adaptation Strategies (with fixed code)
7. **Strategy 1 (Augmentation):** Add real backgrounds to `./data/backgrounds/`, retrain
8. **Strategy 2 (TTA):** Fix RandomHorizontalFlip in strategies.py first, then run
9. **Strategy 3 (DANN):** Run `train_dann.py` starting from new best checkpoint
10. **Strategy 4 (AdaIN):** Run `eval_adain.py` — no training needed

### Report & Presentation
11. Update all results tables with new numbers
12. Generate fresh visualizations
13. Compile final report

---

## 9. PRESENTATION RESULTS TABLE (NEEDS UPDATING)

The old presentation used these numbers (from the buggy training code):

| Method | CD ↓ | F@0.05 ↑ | Params | Notes |
|--------|------|----------|--------|-------|
| Pix2Vox baseline | 0.0911 | 0.1857 | 16.6M | 3D CNN |
| Ours (5-cat base) | 0.0154 | 0.4383 | 17.0M | Old depth GT |
| **Ours (5-cat Cap3D)** | **0.0059** | **0.6807** | 17.0M | Best old code |
| Ours + TTA | 0.0058 | 0.6822 | 17.0M | Marginal |
| Ours + Augmentation | 0.0155 | 0.4333 | 17.0M | Old depth GT |
| Ours + DANN | 0.0157 | 0.4344 | 17.0M | Old depth GT |
| Ours + AdaIN | 0.0329 | — | 17.0M | Hurts synthetic |
| Ours (13-cat) | 0.0161 | 0.4235 | 17.0M | Old code |

**After current retraining finishes**, the 13-cat results should improve significantly due to the flip bug fix + differential LR. Update this table with new numbers.

---

## 10. DATA LAYOUT ON DISK

```
D:\DL\
├── config.yaml                    # Main config (FIXED)
├── config_improved.yaml           # ResNet-34 + 200 epoch config
├── config_augmented.yaml          # Augmentation config
├── model.py                       # Main model
├── dataset.py                     # Main dataset (FIXED — no flip)
├── train.py                       # Training loop (FIXED — diff LR)
├── losses.py                      # Loss functions (evaluate_reconstruction NEEDS CHECK)
├── evaluate.py                    # Evaluation
├── strategies.py                  # TTA/DANN/AdaIN (TTA has flip bug)
├── ...
├── data/
│   ├── shapenet/
│   │   ├── renders/               # 55 categories, symlinked from images
│   │   ├── renders_224/           # Pre-resized to 224×224 JPG
│   │   ├── ShapeNetRendering/     # R2N2 original 13 categories
│   │   └── images/                # Cap3D raw extracted images
│   ├── cap3d/
│   │   └── point_clouds/          # ~39K .npy files (2048×3 each)
│   ├── real_photos/               # Phone photos for qualitative eval
│   └── backgrounds/               # For augmentation (may be empty!)
├── checkpoints/
│   ├── base_pretrained/           # Being overwritten by current run
│   │   ├── best_model.pt
│   │   └── epoch_*.pt
│   ├── cap3d_resnet18_best.pt     # Backup of old best (CD 0.0059)
│   ├── 13cat_best.pt              # Old 13-cat checkpoint
│   ├── dann/best_model.pt         # DANN checkpoint
│   └── improved/                  # ResNet-34 attempt
├── visualizations/
│   ├── synthetic_*.png
│   ├── real_*.png
│   ├── gt_check/
│   ├── adain/
│   └── plots/
├── eval_results/
│   └── results_base_pretrained.json
└── outputs/logs/
    └── base_pretrained/           # TensorBoard logs
```
