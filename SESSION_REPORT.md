# Session Report — March 17, 2026
## Bridging the Synthetic-to-Real Gap in 3D Object Reconstruction
### AI 535 — Mrinal Bharadwaj

---

## 1. BUGS FOUND & FIXED

### Bug 1: `evaluate_reconstruction()` in `losses.py` (CRITICAL)
**Problem:** Called `chamfer_distance(pred, gt, reduce='mean')` — but the function has no `reduce` parameter (it uses `bidirectional=True`). Also unpacked `f_score()` as a tuple `(fs, prec, rec)` — but it returns a single `(B,)` tensor.

**Fix:**
```python
# BEFORE (broken)
cd_loss, cd_p2g, cd_g2p = chamfer_distance(pred, gt, reduce='mean')
fs, prec, rec = f_score(pred, gt, threshold=t)
results[f'f_score@{t}'] = fs

# AFTER (fixed)
cd_loss, cd_p2g, cd_g2p = chamfer_distance(pred, gt, bidirectional=True)
fs = f_score(pred, gt, threshold=t)
results[f'f_score@{t}'] = fs.mean().item()
```

**Impact:** Validation metrics were not being computed correctly. The `best_model.pt` checkpoint only saved at epoch 0 because validation crashed silently.

---

### Bug 2: TTA `RandomHorizontalFlip` in `strategies.py`
**Problem:** `TestTimeAugmentation` class applied `RandomHorizontalFlip` to input images during test-time augmentation. Flipping the image without flipping the predicted point cloud's x-coordinates causes spatial misalignment when averaging predictions.

**Fix:** Removed `transforms.RandomHorizontalFlip(p=0.5)` from the TTA augmentation pipeline.

**Impact:** TTA was partially cancelling itself out — flipped predictions averaged with non-flipped ones produced blurrier point clouds.

---

### Bug 3: `best_model.pt` was epoch 0 (DATA ISSUE)
**Problem:** `checkpoints/base_pretrained/best_model.pt` contained epoch 0 with val_cd=0.024 — NOT the actual best model. The real best was in `epoch_100.pt` with val_cd=0.0059.

**Fix:** Copied `epoch_100.pt` → `best_model.pt`.

**Root cause:** Bug 1 above — validation crashed, so the best-model-saving logic in `train.py` never triggered after epoch 0.

---

### Previously Fixed Bugs (from earlier session)
These were already fixed before this session:

| Bug | What | Fix |
|-----|------|-----|
| RandomHorizontalFlip in training | Flipped images but NOT GT point clouds → 50% misaligned supervision | Removed from `dataset.py` |
| No differential LR | Same LR for pretrained encoder and random decoder → destroyed ImageNet features | Encoder LR = 2e-5, Decoder LR = 5e-4 |
| self_attn_layers = 6 | Too many layers, harder to optimize | Changed to 4 |
| Warmup/Scheduler conflict | Manual warmup conflicted with CosineAnnealingLR | SequentialLR(LinearLR + CosineAnnealingLR) |
| ChamferLoss() constructor | Called with `reduce='mean'` — no such param | `ChamferLoss()` with no args |
| `total_mem` typo | Should be `total_memory` | Fixed |

---

## 2. CURRENT RESULTS

### Synthetic Test (13 categories, 1024 points)
| Metric | Value |
|--------|-------|
| Chamfer Distance | 0.0147 |
| F-Score@0.01 | 0.0194 |
| F-Score@0.02 | 0.1209 |
| F-Score@0.05 | 0.5867 |

### Validation (best checkpoint)
| Metric | Value |
|--------|-------|
| Val CD | 0.0059 |

### All Models Comparison
| Model | Categories | Points | CD ↓ | F@0.05 ↑ | Notes |
|-------|-----------|--------|------|----------|-------|
| Pix2Vox baseline | 5 | 1024 | 0.0911 | 0.1857 | 3D CNN |
| Ours (depth GT) | 5 | 1024 | 0.0154 | 0.4383 | Bad GT data |
| **Ours (Cap3D GT)** | 5 | 1024 | **0.0059** | **0.6807** | Old best |
| Ours + TTA | 5 | 1024 | 0.0058 | 0.6822 | Marginal boost |
| Ours (13-cat, fixed) | 13 | 1024 | 0.0059 val / 0.0147 test | 0.5867 | Bug-fixed code |
| Ours (13-cat, 2048) | 13 | 2048 | *training...* | *training...* | ETA 9 AM PST |

### Real Photo Inference (TTA)
- 27 real photos processed with background removal + TTA
- Model produces recognizable 3D shapes (airplanes, chairs, sofas, cars visible)
- TTA tightens point cloud spread by 3-10% on average
- Background removal is critical — without it, predictions are much worse

---

## 3. WHAT WAS DONE THIS SESSION

1. Read and analyzed `COMPLETE_PROJECT_BRIEFING.md` for full context
2. Fixed `evaluate_reconstruction()` API bugs in `losses.py`
3. Fixed TTA flip bug in `strategies.py`
4. Replaced broken `best_model.pt` with actual best checkpoint
5. Ran VRAM test to determine max batch size (12 for 2048 pts on RTX 4070 Ti Super)
6. Generated TTA inference on all 27 real photos → `visualizations/tta_real/`
7. Generated 20 synthetic test visualizations → `visualizations/synthetic_test/`
8. Added tqdm progress bars + ETA to `train.py`
9. Created `config_2048.yaml` for 2048-point retraining
10. Launched overnight retraining (40 epochs, ~14 hrs, ETA ~9 AM PST March 18)

---

## 4. FILES CREATED/MODIFIED

### Modified
| File | Change |
|------|--------|
| `losses.py` | Fixed `evaluate_reconstruction()` — correct API calls |
| `strategies.py` | Removed `RandomHorizontalFlip` from TTA |
| `train.py` | Added tqdm progress bars, ETA, timedelta logging |

### Created
| File | Purpose |
|------|---------|
| `config_2048.yaml` | Config for 2048-point retraining |
| `launch_train.py` | Training launcher with log file output |
| `training_log.txt` | Live training log (monitor with `tail -f`) |
| `run_tta_real.py` | TTA inference on real photos |
| `gen_synth_viz.py` | Synthetic test visualization generator |
| `quick_eval.py` | Quick evaluation script |
| `vram_test.py` | VRAM usage test for different batch sizes |

### Generated Output
| Path | Contents |
|------|----------|
| `visualizations/tta_real/` | 27 TTA real photo visualizations |
| `visualizations/synthetic_test/` | 20 synthetic pred vs GT visualizations |

---

## 5. OVERNIGHT TRAINING STATUS

**Config:** `config_2048.yaml`
- 2048 query tokens (output points)
- 13 categories, 31,832 training samples
- ResNet-18 encoder (pretrained), 17.5M parameters
- Batch size 12, 40 epochs
- Differential LR: encoder 2e-5, decoder 5e-4
- Mixed precision (FP16)

**Performance:** ~2.1 it/s, ~21 min/epoch, ~14 hours total

**Monitor:** `tail -f D:/DL/training_log.txt`

**Checkpoints:** `D:/DL/checkpoints/retrain_2048/best.pt`

---

## 6. NEXT STEPS (March 18 morning)

1. Check if training finished → look at `training_log.txt` and `checkpoints/retrain_2048/best.pt`
2. Run synthetic test evaluation with new 2048-point model
3. Run real photo inference with TTA using new model
4. Compare old (1024-pt) vs new (2048-pt) results
5. Generate final visualizations for presentation
6. Update results table in presentation
7. **Deadline: 1:30 PM PST**
