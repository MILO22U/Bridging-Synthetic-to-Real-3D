# Lessons Learned — 3D Point Cloud Reconstruction
## AI 535 Deep Learning — Mrinal Bharadwaj

These are hard-earned lessons from debugging this project. Keep these in mind for future DL projects.

---

## 1. Spatial Augmentations Must Be Consistent Across Modalities

**The bug:** `RandomHorizontalFlip` was applied to input images but NOT to ground truth point clouds. This meant 50% of training samples had the image flipped left-right while the GT stayed the same — the model learned to predict symmetric blobs as a compromise.

**The lesson:** Any geometric augmentation (flip, rotation, crop, scale) applied to the input MUST have a corresponding transform on the target. For point clouds:
- Horizontal flip → negate x-coordinate
- Rotation → apply same rotation matrix
- If you can't transform the target, DON'T augment the input spatially

**How to avoid:** Write augmentation as a single function that takes BOTH image and target, transforms them together.

---

## 2. Pretrained Encoders Need Lower Learning Rates

**The bug:** Used the same learning rate (1e-4) for both the pretrained ResNet encoder and the randomly initialized transformer decoder. The encoder's carefully learned ImageNet features got destroyed in the first few epochs.

**The lesson:** Use differential learning rates:
- Pretrained encoder: `base_lr × 0.2` (gentle fine-tuning)
- New decoder layers: `base_lr × 5.0` (fast learning)

**How to avoid:** Always separate params into groups:
```python
optimizer = AdamW([
    {'params': encoder_params, 'lr': base_lr * 0.2},
    {'params': decoder_params, 'lr': base_lr * 5.0},
])
```

---

## 3. Always Verify Loss/Metric Function Signatures

**The bug:** Called `chamfer_distance(pred, gt, reduce='mean')` but the function's signature is `chamfer_distance(pred, target, bidirectional=True)`. The `reduce` kwarg either crashes or gets silently ignored. Also called `fs, prec, rec = f_score(...)` but `f_score` returns a single tensor, not a tuple.

**The lesson:** API signature mismatches can:
- Crash silently during validation → best model never saved
- Produce wrong metric values → misleading results
- Work during training but fail during evaluation

**How to avoid:** Before any training run, test each loss/metric function:
```python
dummy_pred = torch.randn(2, 1024, 3)
dummy_gt = torch.randn(2, 1024, 3)
result = chamfer_distance(dummy_pred, dummy_gt)
print(type(result), result)  # Verify shape and type
```

---

## 4. Always Inspect Checkpoints Before Using Them

**The bug:** `best_model.pt` contained epoch 0 with bad metrics. We assumed it was the best model from 100 epochs of training. The actual best was in `epoch_100.pt`.

**The lesson:** Checkpoint filenames can be misleading. Always inspect:
```python
ckpt = torch.load('best_model.pt', map_location='cpu')
print(f"Epoch: {ckpt['epoch']}, Val CD: {ckpt.get('val_cd', 'N/A')}")
```

**How to avoid:** Save rich metadata in checkpoints (epoch, metrics, config). After training, always verify the best checkpoint is correct.

---

## 5. Test GPU Memory Before Long Training Runs

**The bug:** Tried batch_size=16 with 2048 output points → needed 21GB VRAM on a 16GB card. Discovered only after data loading (which takes minutes).

**The lesson:** Run a quick VRAM test FIRST:
```python
model = MyModel().cuda()
x = torch.randn(batch_size, 3, 224, 224).cuda()
with torch.cuda.amp.autocast():
    pred = model(x)
    pred.sum().backward()
print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
```

**How to avoid:** Create a `vram_test.py` script that tests multiple batch sizes before committing to a training run.

---

## 6. Background Removal is Critical for Synthetic→Real Transfer

**The finding:** The model was trained on synthetic renders with black/uniform backgrounds. Real photos have complex backgrounds (rooms, outdoor scenes) that confuse the encoder.

**The lesson:** The simplest domain adaptation: remove the background from real images at test time (using rembg or similar). This alone dramatically improves real-image predictions.

**How to avoid:** Always consider the input distribution gap. If training data has uniform backgrounds, test-time background removal is the lowest-hanging fruit.

---

## 7. Ground Truth Quality > Model Complexity

**The finding:** Switching from depth-projected point clouds (noisy, misaligned) to Cap3D mesh-sampled point clouds improved CD from 0.0154 to 0.0059 — a 2.6× improvement. No model change needed.

**The lesson:** Bad ground truth = bad model, no matter how sophisticated the architecture. Always verify GT quality visually before training.

**How to avoid:** Visualize 10-20 random GT samples before starting training. If they don't look right, fix the data pipeline first.

---

## Summary Checklist for Future Projects

- [ ] Augmentations consistent across input and target?
- [ ] Differential LR for pretrained vs new layers?
- [ ] Loss/metric API signatures verified with dummy data?
- [ ] Checkpoint metadata includes epoch + metrics?
- [ ] VRAM tested with target batch size before training?
- [ ] Ground truth quality visually verified?
- [ ] Input distribution gap between train and test considered?
