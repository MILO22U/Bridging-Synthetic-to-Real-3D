# Experiment Results — 2048-pt Model
**Generated:** 2026-03-23 07:28

| # | Experiment | Status | Time | Output Dir |
|---|-----------|--------|------|------------|
| 1 | AdaIN alpha=0.3 | FAIL | 1186s | `./visualizations2/adain_alpha0.3` |
| 2 | AdaIN alpha=0.5 | FAIL | 193s | `./visualizations2/adain_alpha0.5` |
| 3 | AdaIN alpha=0.8 | FAIL | 861s | `./visualizations2/adain_alpha0.8` |
| 4 | TTA real photos | PASS | 47s | `./visualizations2/tta_real` |
| 5 | AdaIN VGG eval | PASS | 17024s | `./visualizations2/adain_vgg` |

---

## Checkpoint
- Base model: `checkpoints/retrain_2048/best.pt` (val CD: 0.008105)
- Config: `config_2048.yaml` (2048 pts, 13 categories, ResNet-18)

## Visualization Directories
All outputs in `visualizations2/`:
- `AdaIN_VGG_eval.md` (results)
- `AdaIN_alpha0.3.md` (results)
- `AdaIN_alpha0.5.md` (results)
- `AdaIN_alpha0.8.md` (results)
- `TTA_real_photos.md` (results)
- `adain_alpha0.3/` — 0 files
- `adain_alpha0.5/` — 0 files
- `adain_alpha0.8/` — 0 files
- `adain_alpha03/` — 0 files
- `adain_alpha05/` — 0 files
- `adain_alpha08/` — 0 files
- `adain_vgg/` — 26 files
- `comparison/` — 4 files
- `eval_results/` — 1 files
- `eval_results_tta/` — 1 files
- `real_inference/` — 27 files
- `synthetic_test/` — 30 files
- `tta_real/` — 27 files
