# Bridging the Synthetic-to-Real Gap in 3D Object Reconstruction

**Single-Image 3D Point Cloud Prediction with Domain Adaptation**

Mrinal Bharadwaj | AI 535 — Deep Learning | March 18, 2026

---

## Problem & Motivation

- **Task:** Predict a 3D point cloud from a single RGB image
- Models trained on synthetic data (ShapeNet renders) struggle on real-world photos
- **Domain gap:** lighting, texture, background, camera differences
- **Goal:** Bridge this gap using domain adaptation strategies
- **Key challenge:** No ground-truth 3D for real images → need unsupervised adaptation

---

## Model Architecture

**Hybrid CNN-Transformer (17.5M parameters)**

1. **Encoder:** Pretrained ResNet-18 (ImageNet) → 49 image tokens (7×7 grid)
2. **Cross-Attention** (2 layers, 8 heads): 2048 learnable query tokens attend to image features
3. **Self-Attention Decoder** (4 layers, 8 heads): refine query representations
4. **MLP Head:** project each query to (x, y, z) → 2048-point cloud

**Training details:**
- Differential LR: encoder 0.2× base, decoder 5× base
- Cosine LR schedule with 2-epoch warmup
- Mixed precision training (FP16)
- Loss: Chamfer Distance (bidirectional)

---

## Training Setup

- **Dataset:** ShapeNet + Cap3D renders (13 categories, 31,832 training samples)
- **Categories:** airplane, bench, cabinet, car, chair, display, lamp, loudspeaker, rifle, sofa, table, telephone, watercraft
- **Output:** 2048-point 3D point cloud per image
- **Batch size:** 12 | **Epochs:** 40 | **GPU:** RTX 4070 Ti Super (16GB)
- **Augmentation:** color jitter, gaussian blur, random crop, random erasing

**Key bug fixes applied before training:**
- Flip augmentation: negate point cloud x-coords when image is flipped
- Differential learning rates for pretrained encoder vs decoder
- ChamferLoss API signature correction

---

## Training Curves

- **Training loss:** Converges smoothly from 0.30 → 0.006 over 40 epochs
- **Validation CD:** Best = 0.00930 at epoch 40, steady improvement throughout
- Model was still improving at end of training → more epochs would help

*See: `visualizations/presentation/training_loss_curve.png`, `val_chamfer_distance.png`*

---

## Quantitative Results

| Metric | Old (1024pts, 100ep) | New (2048pts, 40ep) | Improvement |
|---|---|---|---|
| **CD Mean** | 0.01522 | **0.00980** | -35.6% |
| **F-Score@0.01** | 0.0196 | **0.0424** | +116% |
| **F-Score@0.02** | 0.1211 | **0.1958** | +61.7% |
| **F-Score@0.05** | 0.5865 | **0.6615** | +12.8% |

New 2048-pt model significantly outperforms old 1024-pt model on all metrics despite training for fewer epochs.

*See: `visualizations/presentation/results_comparison.png`*

---

## Synthetic Test Results — Best Samples

Model captures overall shape well for compact objects:
- **Airplanes:** elongated body + wings clearly predicted
- **Cars:** body shape and proportions match GT
- **Rifles:** thin elongated structures captured
- **Lamps:** spherical heads reconstructed
- **Cabinets:** dense box-like shapes

Predicted point clouds (blue) closely match ground truth (green).

*See: `synth_airplane.png`, `synth_car1.png`, `synth_car2.png`, `synth_rifle1.png`, `synth_rifle2.png`, `synth_lamp.png`, `synth_cabinet.png`*

---

## Domain Adaptation: Test-Time Augmentation (TTA)

**Strategy:**
- Apply 10 random augmentations at inference, average predictions
- Augmentations: color jitter, random crop, horizontal flip (with x-coord correction)
- No retraining needed — applied at test time only
- Reduces prediction variance (spread), producing tighter point clouds

**Results on 27 real photos:**
- TTA consistently reduces point cloud spread across all images
- Blue = No TTA (scattered) | Orange = With TTA (tighter, more coherent)

*See: `real_airplane.png`, `real_car.png`, `real_monitor.png`, `real_lamp.png`, `real_rifle1.png`, `real_rifle2.png`*

---

## Real-World Inference Results

Model generalizes from synthetic training to real photographs:
- **Car (BMW):** compact car shape predicted from internet photo
- **Monitor:** flat panel display shape captured
- **Desk lamp:** lamp shape with base and head
- **Rifles:** recognizable weapon shapes from real photos

Background removal (rembg) + TTA significantly improves reconstruction quality.

---

## Qualitative Comparison: 1024-pt vs 2048-pt on Chairs

**Old Model (1024 pts, 100 epochs):**
- Better chair structure — legs, seat, back visible
- 100 epochs gave more time to learn fine details

**New Model (2048 pts, 40 epochs):**
- Denser predictions but more scattered on thin structures
- Better overall metrics on compact shapes
- With more training epochs, would likely surpass old model on chairs too

*See: `old_chair1.png` through `old_chair4.png` vs `limitation_chair1.png`, `limitation_chair2.png`*

---

## Limitations

- **Thin structures** (chair legs, lamp arms) remain challenging
- **Chamfer Distance** allows spread-out solutions — doesn't penalize off-surface points
- **Single-view ambiguity:** occluded parts are guessed, not seen
- 2048-pt model only trained 40 epochs (vs 100 for 1024-pt baseline)

## Future Work

- Train 2048-pt model for 100+ epochs for fair comparison
- Add **Earth Mover's Distance (EMD)** or surface-aware loss for tighter reconstructions
- **Multi-view fusion:** combine predictions from multiple viewpoints
- **AdaIN style transfer** for stronger domain adaptation
- **Mesh reconstruction** from predicted point clouds

---

## Conclusion

- Built a **hybrid CNN-Transformer** model for single-image 3D point cloud reconstruction
- Trained on **13 ShapeNet categories** (31,832 samples) with 2048-point output
- Achieved **35.6% lower Chamfer Distance** and **2× better F-Score@0.01** vs baseline
- **Test-Time Augmentation** successfully bridges synthetic-to-real domain gap without any real-world 3D supervision
- Model **generalizes to real photographs** across multiple object categories
- Identified key training pitfalls: flip augmentation alignment, differential LR, and loss function API verification are critical for point cloud reconstruction

---

## Thank You! Questions?

Mrinal Bharadwaj | AI 535 | March 2026
