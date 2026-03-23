# Experiment Log — Bridging Synthetic-to-Real Gap in 3D Reconstruction
### AI 535 Deep Learning | Mrinal Bharadwaj | Oregon State University

---

## Day 0 — Early March 2026
### Phase: Data Acquisition & Preprocessing

**What was done:**
- Downloaded ShapeNet R2N2 renderings (13 categories, ~45K objects, 24 views each)
- Downloaded Cap3D rendered images (8-view renders + depth maps) from HuggingFace
- Ran `download_shapenet_renders.py`, `restructure_renders.py` to organize into `renders/<synset>/<model_id>/image_XXXX.png`
- Ran `preresize_images.py` to resize 512→224px for training speed → `renders_224/`
- Attempted to download `Cap3D_pcs.npz` — **failed** (file was only 15 bytes)
- Fallback: generated point clouds from depth maps via `generate_pointclouds.py`

**What was observed:**
- Depth-projected point clouds were **fundamentally broken** — multi-view back-projection had camera extrinsic alignment issues, resulting in scattered, disconnected clusters instead of solid shapes
- This bad GT would go on to poison early training runs (see Day 1)

**Next step:** Train with available data to establish a baseline, investigate GT quality

---

## Day 1 — Early March 2026
### Phase: First Training Runs (Depth-Projected GT)

**What was done:**
- Training Run 1: 5 categories, depth-projected GT → early stopped at epoch 37
- Training Run 2: 5 categories, depth-projected GT, lower LR → also early stopped

**Results:**
| Run | CD ↓ | F@0.05 ↑ | Notes |
|-----|------|----------|-------|
| Run 1 | 0.1767 | 0.0221 | Terrible — GT was garbage |
| Run 2 | 0.0154 | 0.4383 | Better, still limited by bad GT |

**What was observed:**
- Even with lower LR, performance ceiling was clear — model couldn't learn accurate shapes from scattered GT clusters
- Confirmed the depth-projected ground truth was the bottleneck, not the model

**Next step:** Find proper mesh-sampled ground truth point clouds

---

## Day 2 — Mid March 2026
### Phase: Fixed Ground Truth with Cap3D PLY Files

**What was done:**
- Successfully downloaded Cap3D PLY point clouds (52,472 objects, 16,384 pts each)
- Wrote `convert_cap3d_ply_v2.py` to handle ASCII PLY format → NPY (2048×3 each)
- Verified quality with `check_pointclouds.py` — clean, solid 3D shapes
- Sub-sampled 16,384 → 2,048 points per object

**What changed:**
- Replaced all depth-projected GT with proper mesh-sampled point clouds
- This was the **single biggest improvement** in the entire project — 2.6× CD improvement with zero model changes (see Day 3)

**Next step:** Retrain model with the clean GT

---

## Day 3 — ~March 14–15, 2026
### Phase: Retrained with Cap3D GT + Pix2Vox Baseline

**What was done:**
- Training Run 3: 5 categories, Cap3D GT, ResNet-18, 100 epochs
- Training Run 4: 13 categories, Cap3D GT, ResNet-18, 100 epochs (31,832 samples)
- Implemented `pix2vox_baseline.py` (3D CNN baseline) for comparison

**Results:**
| Model | CD ↓ | F@0.05 ↑ | Notes |
|-------|------|----------|-------|
| Pix2Vox baseline | 0.0911 | 0.1857 | 3D CNN (16.6M params) |
| Ours (5-cat, Cap3D) | **0.0059** | **0.6807** | Best result (epoch 85) |
| Ours (13-cat, Cap3D) | 0.0161 | 0.4235 | Harder problem |

**What was observed:**
- Our transformer model is **6× better** than Pix2Vox on CD
- 13-category model underperformed vs 5-cat — expected with more diverse shapes
- Checkpoints: `cap3d_resnet18_best.pt` (5-cat), `13cat_best.pt` (13-cat)

**What changed (ref Day 1):**
- CD improved from 0.0154 → 0.0059 purely from GT quality (2.6× improvement)

**Next step:** Domain adaptation experiments to bridge synthetic→real gap

---

## Day 4 — ~March 15–16, 2026
### Phase: Domain Adaptation Experiments (OLD Code — Had Bugs)

**What was done (all with OLD code that had undetected bugs):**
1. **Training-Time Augmentation** — heavy color jitter, blur, random erasing → CD 0.0155, F@0.05 0.4333
2. **Test-Time Augmentation (TTA)** — 10 augmented views averaged → CD 0.0058, F@0.05 0.6822
3. **DANN** — domain adversarial fine-tuning with real images → CD 0.0157, F@0.05 0.4344
4. **AdaIN Style Transfer** — VGG feature statistics matching → CD 0.0329 (hurts synthetic)

**What was observed:**
- TTA gave marginal free improvement (no retraining needed)
- Training-time augmentation and DANN showed no real gain over base model
- AdaIN hurt synthetic performance (expected — designed for real-to-synthetic transfer only)
- **Unknown at this time:** all results were contaminated by flip augmentation bug (see Day 6)

**Next step:** Try larger encoder (ResNet-34) + longer training

---

## Day 5 — March 16, 2026
### Phase: ResNet-34 + 200 Epoch Attempt

**What was done:**
- Created `config_improved.yaml` with ResNet-34 encoder, 200 epochs, 13 categories
- Model: 29.2M parameters (vs 17.5M with ResNet-18)
- Started training

**What was observed:**
- Training was **interrupted** by the bug-fix session (Day 6)
- Partial checkpoint saved in `checkpoints/improved/`

**Next step:** This run was abandoned — bug fixes took priority

---

## Day 6 — March 17, 2026
### Phase: Bug Discovery & Fix Session (CRITICAL)

**Trigger:** User uploaded real photo inference results showing model producing **diffuse blob predictions**. Investigation revealed **7 bugs** in the training pipeline.

**Bugs Found and Fixed:**

| # | Severity | File | Bug | Fix |
|---|----------|------|-----|-----|
| 1 | CRITICAL | `dataset.py:77` | `RandomHorizontalFlip` flipped images but NOT GT point clouds — 50% misaligned supervision | Removed flip from transforms |
| 2 | CRITICAL | `train.py:256` | Same LR (1e-4) for pretrained encoder AND random decoder — destroyed ImageNet features | Differential LR: encoder 0.2×, decoder 5× |
| 3 | Major | `config.yaml:18` | `self_attn_layers=6` (too many, guide said 4) — optimization difficulty | Changed to 4 layers |
| 4 | Major | `train.py:303` | Manual warmup conflicted with CosineAnnealingLR | `SequentialLR(LinearLR + CosineAnnealingLR)` |
| 5 | Major | `losses.py` | `chamfer_distance(pred, gt, reduce='mean')` — wrong API; `f_score()` return type wrong | Correct API: `bidirectional=True`, `.mean().item()` |
| 6 | Minor | `losses.py` | `ChamferLoss(reduce='mean')` — no such param | `ChamferLoss()` with no args |
| 7 | Minor | `train.py` | `total_mem` typo → should be `total_memory` | Fixed attribute name |

**Critical Discovery:**
- `checkpoints/base_pretrained/best_model.pt` contained **epoch 0** data (val_cd=0.024), not the actual best
- Root cause: Bug 5 crashed validation silently → best-model-saving never triggered past epoch 0
- Fix: Copied `epoch_100.pt` → `best_model.pt` (actual val_cd=0.0059)

**What changed (ref Day 4):**
- All prior domain adaptation results were contaminated by these bugs
- The flip bug alone caused the model to learn symmetric blobs as a compromise
- All experiments would need to be rerun with fixed code

**Next step:** Retrain from scratch with all fixes, 2048-pt output

---

## Day 7 — March 17–18, 2026
### Phase: 2048-Point Retraining Begins

**What was done:**
- Created `config_2048.yaml` — doubled output from 1024 → 2048 points
- Ran `vram_test.py` — determined max batch_size=12 for 2048 pts on 16GB RTX 4070 Ti Super
- Created `launch_train.py` for log file output
- Added tqdm progress bars + ETA to `train.py`
- Started training: 13 categories, 31,832 samples, 100 epochs

**Training config:**
```
num_query_tokens: 2048, batch_size: 12, lr: 1e-4
Encoder LR: 2e-5 (×0.2), Decoder LR: 5e-4 (×5.0)
Scheduler: LinearLR warmup (2 ep) + CosineAnnealingLR (98 ep)
Mixed precision FP16, gradient clipping max_norm=1.0
```

**Convergence (first 40 epochs):**
| Epoch | Val CD | Δ |
|-------|--------|---|
| 1 | 0.097805 | — |
| 3 | 0.023438 | −76% |
| 6 | 0.014291 | −39% |
| 10 | 0.012392 | −13% |
| 17 | 0.010851 | −12% |
| 27 | 0.009838 | −9% |
| 40 | 0.009296 | −6% |

**What was observed:**
- Training speed: ~2.1 it/s, ~21 min/epoch
- Rapid convergence in first 10 epochs, then diminishing returns
- Epoch 40 already competitive: CD 0.009296, F@0.05 0.6642

**Next step:** Resume training to 100 epochs

---

## Day 8 — March 19–22, 2026
### Phase: Resume Training to 100 Epochs

**What was done:**
- Fixed scheduler resume logic: fast-forward cosine scheduler state on checkpoint load
- Added file logging for PowerShell monitoring (`Get-Content -Tail -Wait`)
- Resumed from epoch 40 checkpoint, trained epochs 41–100
- Training completed: **March 22, 2026 at 21:12**

**Final Results (ref Day 7 for comparison):**
```
Best Val CD:   0.008105 (epoch ~98)
Test CD:       0.008560
F@0.01:        0.0448
F@0.02:        0.2068
F@0.05:        0.6858
```

**What changed (ref Day 3):**
- F@0.01 doubled (0.0223 → 0.0448) — much better fine-detail reconstruction
- F@0.02 improved 44% (0.1428 → 0.2068)
- F@0.05 roughly matched (0.6822 → 0.6858)
- CD numerically higher (0.00582 → 0.00856) — expected: 2× more points = larger coverage area

**Checkpoint:** `checkpoints/retrain_2048/best.pt` (epoch ~98, val CD 0.008105)

**Next step:** Rerun all domain adaptation experiments with fixed code and 2048-pt model

---

## Day 9 — March 23, 2026 (AM)
### Phase: Full Experiment Rerun with Fixed Code

**What was done:**
- Generated `visualizations2/` with 30 synthetic test samples, 27 real photo inferences, comparison charts
- Ran baseline evaluation (no TTA): CD 0.00858, F@0.05 0.6829
- Ran TTA evaluation: CD 0.00917, F@0.05 0.6761
- Launched `run_all_experiments.py` — sequential pipeline:
  1. AdaIN alpha=0.3
  2. AdaIN alpha=0.5
  3. AdaIN alpha=0.8
  4. TTA on real photos
  5. AdaIN VGG evaluation
  6. DANN training (20 epochs)

**Results so far:**

| Experiment | Status | Time | Key Finding |
|-----------|--------|------|-------------|
| Baseline (no TTA) | PASS | — | CD 0.00858, F@0.05 0.6829 |
| TTA (synthetic) | PASS | — | CD 0.00917, F@0.05 0.6761 — TTA **hurts** 2048-pt model |
| AdaIN α=0.3 | **FAIL** | 1186s | OOM at inference sample 10/27 |
| AdaIN α=0.5 | **FAIL** | 193s | OOM at inference sample 10/27 |
| AdaIN α=0.8 | **FAIL** | 861s | OOM at inference sample 10/27 |
| TTA real photos | PASS | 47s | 27/27 processed; spread reduced on 8/10 samples |
| AdaIN VGG eval | RUNNING | — | Processing 3979 test samples (batch_size=1) |
| DANN training | QUEUED | — | 20 epochs, batch_size=8 |

**What was observed:**
- TTA slightly hurts the 2048-pt model (opposite of 1024-pt behavior) — the larger model is already well-calibrated and averaging augmented views adds noise
- AdaIN OOM root cause: `create_dataloaders()` loads full dataset with `num_workers=4`, each worker forks process memory, while model+VGG are on GPU simultaneously
- TTA on real photos reduces prediction spread in most cases (8/10 shown), indicating more stable predictions

**What changed:**
- Applied OOM fix to `run_adain.py`: compute stats on CPU first → delete dataloader entirely → then load model for inference
- Created `PROJECT_COMPLETE_HISTORY.md` and `report_package/main.tex` (CVPR-format LaTeX report)

**TTA Real Photos — Spread Analysis (subset):**
| Photo | Plain Spread | TTA Spread | Δ |
|-------|-------------|-----------|---|
| desk-lamp | 0.3113 | 0.1982 | −36% |
| airplane (blue sky) | 0.2770 | 0.2325 | −16% |
| car (red) | 0.2213 | 0.2048 | −7% |
| wooden-chair | 0.2158 | 0.1959 | −9% |
| monitor (old) | 0.2527 | 0.2660 | +5% (worse) |

**Next step:** Wait for AdaIN VGG eval to finish → DANN training → rerun fixed AdaIN → regenerate comparison charts

---

## Day 9 — March 23, 2026 (Current — 06:18 AM)
### Phase: Waiting on Long-Running Evaluations

**What is running:**
- `eval_adain.py` (PID 11808) — started 02:44, running ~3.5 hours
  - Processing 3,979 synthetic test samples, batch_size=1
  - Each sample: original model forward + VGG style transfer + styled model forward + 2× Chamfer Distance
  - GPU: 100% utilization, 15.8/16.4 GB VRAM
  - Estimated completion: **imminent** (~3.3h expected total)

**What remains after current task:**
| Task | Est. Time | Status |
|------|-----------|--------|
| DANN training (20 epochs) | 3–5 hours | Queued next |
| AdaIN x3 rerun (OOM-fixed) | ~30 min | Queued |
| Comparison chart regen | ~2 min | Queued |
| Report/presentation update | Manual | Pending final numbers |

**Estimated total remaining: ~4–6 hours**

**Next step:** Monitor AdaIN VGG eval completion → DANN training auto-starts → rerun fixed AdaIN experiments → compile final results table
