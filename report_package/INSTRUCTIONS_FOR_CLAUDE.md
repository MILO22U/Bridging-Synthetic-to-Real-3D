# Instructions: Add These Images to the LaTeX Report

You are given `main.tex`, `references.bib`, and 15 images in the `figures/` folder. Insert all figures into the report using the exact LaTeX code below. The report uses the CVPR 2016 template and MUST stay within 4-8 pages.

---

## Figure 2: Training Curves
**Location:** After Section 3.2 (Training), after the Data Augmentation paragraph
**Images:** `training_loss_curve.png` + `val_chamfer_distance.png`

| File | Description |
|------|-------------|
| `figures/training_loss_curve.png` | Training loss over steps, converges from 0.30 to 0.006 |
| `figures/val_chamfer_distance.png` | Validation Chamfer Distance over epochs, best 0.00930 at epoch 40 |

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.48\linewidth]{figures/training_loss_curve.png}
\hfill
\includegraphics[width=0.48\linewidth]{figures/val_chamfer_distance.png}
\caption{Training curves. \textbf{Left:} Training loss converges smoothly from 0.30 to 0.006. \textbf{Right:} Validation Chamfer Distance reaches 0.00930 at epoch 40, with the model still improving.}
\label{fig:training_curves}
\end{figure}
```

---

## Figure 3: Synthetic Test Results (FULL-WIDTH)
**Location:** Start of Section 4.5 (Qualitative Results)
**Images:** 4 synthetic reconstructions — each shows input image, predicted (blue), ground truth (green), side view

| File | Category |
|------|----------|
| `figures/synth_airplane.png` | Airplane — elongated fuselage shape, good reconstruction |
| `figures/synth_car1.png` | Car — sports car, dense point cloud matches GT well |
| `figures/synth_lamp.png` | Lamp — ceiling/pendant lamp, spherical shape captured |
| `figures/synth_cabinet.png` | Cabinet — rectangular box shape, dense coverage |

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=0.24\textwidth]{figures/synth_airplane.png}
\hfill
\includegraphics[width=0.24\textwidth]{figures/synth_car1.png}
\hfill
\includegraphics[width=0.24\textwidth]{figures/synth_lamp.png}
\hfill
\includegraphics[width=0.24\textwidth]{figures/synth_cabinet.png}
\caption{Synthetic test results. Each panel shows input image, predicted point cloud (blue), ground truth (green), and side view. From left to right: airplane, car, lamp, cabinet.}
\label{fig:synth_results}
\end{figure*}
```

---

## Figure 4: Real-World Inference Results (FULL-WIDTH)
**Location:** After Figure 3 in Section 4.5
**Images:** 4 real-world photos — each shows original photo, background removed, No TTA prediction, With TTA prediction

| File | Subject |
|------|---------|
| `figures/real_car.png` | Red BMW car from internet — recognizable car shape |
| `figures/real_monitor.png` | Computer monitor with Windows wallpaper — flat screen shape |
| `figures/real_lamp.png` | IKEA-style green desk lamp — lamp form captured |
| `figures/real_rifle2.png` | Sniper rifle on rocks — elongated rifle shape |

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=0.24\textwidth]{figures/real_car.png}
\hfill
\includegraphics[width=0.24\textwidth]{figures/real_monitor.png}
\hfill
\includegraphics[width=0.24\textwidth]{figures/real_lamp.png}
\hfill
\includegraphics[width=0.24\textwidth]{figures/real_rifle2.png}
\caption{Real-world inference on internet photographs. From left to right: car, monitor, desk lamp, rifle. Background removal + TTA produces recognizable 3D shapes from single images.}
\label{fig:real_results}
\end{figure*}
```

---

## Figure 5: TTA Before/After Effect
**Location:** Section 4.3 (Domain Adaptation Analysis), after the TTA paragraph
**Images:** 2 real-world examples showing No TTA (blue, scattered) vs With TTA (orange, tighter)

| File | Subject |
|------|---------|
| `figures/real_airplane.png` | Commercial airplane — blue (no TTA) is scattered, orange (TTA) is tighter and more coherent |
| `figures/real_rifle1.png` | AK-47 rifle — blue (no TTA) is noisy, orange (TTA) produces cleaner elongated shape |

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.48\linewidth]{figures/real_airplane.png}
\hfill
\includegraphics[width=0.48\linewidth]{figures/real_rifle1.png}
\caption{Test-Time Augmentation effect. Blue = without TTA (scattered), orange = with TTA (tighter, more coherent). Left: airplane. Right: rifle.}
\label{fig:tta_effect}
\end{figure}
```

---

## Figure 6: Chair Comparison (Limitation Analysis)
**Location:** Section 4.5 after qualitative text, OR in Section 5 (Discussion/Limitations)
**Images:** Old model vs New model on chairs — shows trade-off between training duration and point density

| File | Description |
|------|-------------|
| `figures/old_chair1.png` | Old model (1024 pts, 100 epochs) — blue prediction captures chair legs and structure clearly, CD shown in title |
| `figures/limitation_chair1.png` | New model (2048 pts, 40 epochs) — orange prediction is denser but more scattered on thin structures like legs, CD shown in title |

NOTE: These are comparison images showing Input Image, Old model (blue), New model (orange), Ground Truth (green), and New side view. Use the full images and reference the blue vs orange colors in the caption.

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.48\linewidth]{figures/old_chair1.png}
\hfill
\includegraphics[width=0.48\linewidth]{figures/limitation_chair1.png}
\caption{Chair comparison. \textbf{Left:} Old model (1024 pts, 100 epochs) captures legs and structure (blue). \textbf{Right:} New model (2048 pts, 40 epochs) is denser but scattered on thin structures (orange).}
\label{fig:chair_comparison}
\end{figure}
```

---

## Figure 7: Results Comparison Bar Chart
**Location:** Section 4.2, next to or after Table 2
**Image:** Bar chart comparing old (1024-pt, 100 epochs) vs new (2048-pt, 40 epochs) model

| File | Description |
|------|-------------|
| `figures/results_comparison.png` | Side-by-side bar chart of Chamfer Distance and F-Score metrics for both models |

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.95\linewidth]{figures/results_comparison.png}
\caption{Visual comparison of old (1024-pt, 100 epochs) vs new (2048-pt, 40 epochs) model across Chamfer Distance and F-Score metrics.}
\label{fig:results_bar}
\end{figure}
```

---

## Text References to Add

Add these cross-references in the appropriate sections:

- **Training section:** "Training curves are shown in Figure~\ref{fig:training_curves}."
- **Main results (Sec 4.2):** "A visual comparison is shown in Figure~\ref{fig:results_bar}."
- **TTA analysis (Sec 4.3):** "The effect of TTA is visualized in Figure~\ref{fig:tta_effect}."
- **Qualitative results (Sec 4.5):** Reference "Figure~\ref{fig:synth_results}", "Figure~\ref{fig:real_results}", and "Figure~\ref{fig:chair_comparison}"

---

## Page Limit (4-8 pages)

If the report exceeds 8 pages after adding all figures:
1. **First:** Change `figure*` to `figure` for Figures 3 and 4 (single column instead of full-width)
2. **Second:** Remove Figure 7 (bar chart) since Table 2 already has the same data
3. **Third:** Combine Figures 5 and 6 into one figure with subfigures

---

## Compilation

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
