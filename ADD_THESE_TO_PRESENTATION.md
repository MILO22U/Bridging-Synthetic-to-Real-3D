# Missing Content — Add to AI535_3D_Reconstruction_Presentation.pptx

The redesigned presentation is missing several slides/content from the original `AI535_Final_Presentation.pptx`. Add the following:

---

## 1. NEW SLIDE: Full Results Table (after Training Curves slide)

Add a results table that includes the Pix2Vox baseline and all strategies:

| Method | Chamfer Distance ↓ | F-Score@0.05 ↑ | Notes |
|---|---|---|---|
| Pix2Vox Baseline | 0.0911 | 0.1857 | 3D CNN approach |
| Ours (5 categories) | 0.0154 | 0.4383 | Best base model |
| Ours + TTA | 0.0153 | 0.4400 | Best overall |
| Ours + Augmentation | 0.0155 | 0.4333 | Strategy 1 |
| Ours + DANN | 0.0157 | 0.4344 | Strategy 3 |
| Ours + AdaIN | 0.0329 | — | Strategy 4 (hurts synthetic) |
| Ours (13 categories) | 0.0161 | 0.4235 | Full ShapeNet |

Include a big callout stat: **6× lower Chamfer Distance than Pix2Vox** with similar parameter count (17M).

---

## 2. NEW SLIDE: Domain Adaptation Strategies Comparison (after TTA slide)

Show all 4 strategies side by side as cards/columns:

**Strategy 1 — Training-Time Augmentation**
- Heavy color jitter, blur, random erasing during training
- Slight regularization effect
- CD: 0.0155

**Strategy 2 — Test-Time Augmentation (TTA)**
- 10 augmented views averaged at inference
- Best result (free improvement)
- CD: 0.0153 ✓ (mark as winner)

**Strategy 3 — DANN**
- Domain adversarial training with real images
- Domain-invariant features learned
- CD: 0.0157

**Strategy 4 — AdaIN Style Transfer**
- Transform real images to synthetic style
- Hurts synthetic performance (designed for real only)
- CD: 0.0329

---

## 3. NEW SLIDE: Key Findings (before Limitations slide)

2×2 grid layout with these 4 findings:

**Pretrained > Scratch**
- ImageNet-pretrained encoder dramatically improves convergence and feature quality
- Real-world pretraining inherently reduces the domain gap

**Transformer > CNN for 3D**
- Cross-attention + self-attention (CD: 0.015) vastly outperforms 3D CNN decoder (Pix2Vox CD: 0.091)
- Same encoder and parameter count

**More Categories = Harder**
- 5 categories (CD: 0.015) vs 13 categories (CD: 0.016)
- Model spreads capacity across diverse shapes, but still performs well

**TTA is Free & Effective**
- Test-time augmentation gives consistent improvement with zero retraining cost
- Best overall results

---

## 4. UPDATE: Conclusion/Key Takeaways slide

Add these missing points to the conclusion:
- Studied **4 domain adaptation strategies** with systematic comparison
- Achieved **6× better Chamfer Distance than Pix2Vox baseline**
- Demonstrated that **pretrained encoders inherently reduce domain gap**

---

## Summary of changes needed:
1. Add full results table slide (with Pix2Vox baseline)
2. Add 4-strategy comparison slide
3. Add key findings slide (2×2 grid)
4. Update conclusion with baseline comparison and strategy study mentions
5. Update slide numbers accordingly (total will go from 14 to 17)
