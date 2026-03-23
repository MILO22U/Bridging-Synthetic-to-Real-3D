# EASU Model Diagnostic Report: Why Your Predictions Are Blobs

## Summary

After reviewing your entire codebase, I found **3 critical bugs**, **2 major configuration issues**, and **several minor improvements** that together explain why your model produces diffuse point clouds on real images (and likely performs poorly even on synthetic test data).

---

## CRITICAL BUG #1: RandomHorizontalFlip Breaks Point Cloud Alignment

**File:** `dataset.py`, line 77  
**Severity:** 🔴 Critical — This alone can prevent the model from learning

Your training transform includes `T.RandomHorizontalFlip(p=0.5)`, but the corresponding ground truth point cloud is **never flipped**. This means 50% of the time during training, the model sees a **left-right flipped image** paired with a **non-flipped point cloud**. This creates contradictory supervision: the model literally learns conflicting things half the time.

**Example:** If the training image shows a car facing left (after flip), but the GT point cloud has the car facing right — the model tries to average these out, producing a symmetric blob.

**Fix:** Either remove `RandomHorizontalFlip` entirely, or negate the x-coordinate of the GT point cloud when the image is flipped. Since flipping is hard to coordinate in a `transforms.Compose` pipeline, the simplest fix is to **remove it**:

```python
# In get_train_transform(), REMOVE this line:
T.RandomHorizontalFlip(p=0.5),  # DELETE THIS
```

---

## CRITICAL BUG #2: No Differential Learning Rate

**File:** `train.py`, lines 256-259  
**Severity:** 🔴 Critical — Pretrained encoder gets disrupted too quickly

Your STEP_BY_STEP_GUIDE specifically says:
> "Optimizer: AdamW, lr=1e-4 (encoder), lr=5e-4 (decoder) — differential LR"

But `train.py` uses a **single learning rate** for everything:

```python
optimizer = optim.AdamW(
    model.parameters(),  # ALL params get the same lr!
    lr=cfg['training']['learning_rate'],  # 1e-4
    ...
)
```

This means the pretrained ResNet-18 encoder (which already has good features from ImageNet) is being updated at the same rate as the randomly-initialized decoder. The encoder's pretrained features get destroyed before the decoder learns to use them.

**Fix:** Use parameter groups with different learning rates:

```python
encoder_params = list(model.encoder.parameters())
encoder_param_ids = set(id(p) for p in encoder_params)
decoder_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]

optimizer = optim.AdamW([
    {'params': encoder_params, 'lr': cfg['training']['learning_rate'] * 0.2},  # 2e-5
    {'params': decoder_params, 'lr': cfg['training']['learning_rate'] * 5},    # 5e-4
], weight_decay=cfg['training']['weight_decay'])
```

---

## CRITICAL BUG #3: Self-Attention on 2048 Tokens × 6 Layers Is Too Heavy

**File:** `config.yaml`, line 18  
**Severity:** 🟠 Major — Causes optimization difficulty and likely underfitting

Your config has `self_attn_layers: 6`, but your guide recommends 4. Self-attention on 2048 tokens means each layer computes a 2048×2048 attention matrix. With 6 layers this is:
- Very memory-intensive
- Creates vanishing/exploding gradient risks
- The decoder has far more capacity than the cross-attention bridge (2 layers), creating a bottleneck

**Fix:** Reduce to 4 self-attention layers:

```yaml
self_attn_layers: 4    # was 6
```

---

## MAJOR ISSUE #4: Train Transform Pipeline Order

**File:** `dataset.py`, lines 68-93

The transform pipeline does `Resize(224) → RandomResizedCrop(224)`. This is slightly wasteful but more importantly, the `RandomResizedCrop` with scale `(0.8, 1.0)` can crop away parts of the object. Combined with Bug #1 (the flip), this further degrades the alignment between images and point clouds.

The crop itself is fine for domain robustness, but paired with the flip bug, it amplifies the damage.

---

## MAJOR ISSUE #5: No Learning Rate Warmup for Scheduler Interaction

**File:** `train.py`, lines 303-306 and 318-319

Your warmup manually overrides `lr` for the first 5 epochs, but then calls `scheduler.step()` starting at epoch 5. The `CosineAnnealingLR` scheduler doesn't know about the warmup — it starts its cosine decay from epoch 0 internally. So at epoch 5, the scheduler's internal counter is already at step 5, and the LR drops prematurely.

**Fix:** Use `torch.optim.lr_scheduler.SequentialLR` or `LinearLR` for warmup:

```python
warmup_scheduler = optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=warmup_epochs
)
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg['training']['num_epochs'] - warmup_epochs,
    eta_min=cfg['training']['learning_rate'] * 0.01,
)
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_epochs]
)
# Then just call scheduler.step() every epoch, no manual warmup logic needed
```

---

## MINOR ISSUES

### 6. Chamfer Distance uses squared distances (not a bug, but know this)
Your `chamfer_distance()` computes squared L2 distances. This is standard for training (smoother gradients), but your reported CD values will be in squared units. When comparing to papers, some report `sqrt(CD)` — just be aware of this.

### 7. TTA implementation is fake
In `evaluate.py`, the TTA just adds Gaussian noise to already-normalized tensors:
```python
noise = torch.randn_like(images_raw) * 0.02
aug_images = (images_raw + noise).clamp(0, 1)
```
This doesn't do real augmentation. You already have proper `get_tta_transforms()` in `dataset.py` — use those instead by running inference with each transform on the raw PIL images.

### 8. Background replacement may not fire
`use_random_backgrounds: true` in config, but if `./data/backgrounds` directory is empty or missing, it silently does nothing. Verify you actually have background images there.

---

## Recommended Fix Priority

1. **Remove RandomHorizontalFlip** ← Do this first, retrain, see if loss drops
2. **Add differential learning rates** ← Pretrained features survive training
3. **Reduce self_attn_layers to 4** ← Faster training, better optimization
4. **Fix warmup/scheduler interaction** ← Proper LR schedule
5. Verify background augmentation works ← Check the data directory
6. Fix TTA for evaluation ← Better real-world numbers

---

## What Good Metrics Should Look Like

For reference, on ShapeNet→Cap3D with this architecture:
- **Synthetic Val CD:** ~0.01-0.03 (squared) after 50 epochs
- **F-Score@0.05:** > 0.3 on synthetic test
- **Real-world CD:** 1.5-3× higher than synthetic (this IS the domain gap)

If your synthetic CD is > 0.1, the model hasn't learned meaningful shapes yet.
