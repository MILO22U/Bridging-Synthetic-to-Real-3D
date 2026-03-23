# Instructions: Add Experiment Images to LaTeX Report

## Files You Have

The report is in the CVPR 2016 LaTeX template. The main file is `main.tex` with `references.bib`. Style files: `cvpr.sty`, `eso-pic.sty`, `cvpr_eso.sty`, `ieee.bst`.

## Images to Add

All images are in the `figures/` folder. Here's what each one is:

### Training Curves
- `figures/training_loss_curve.png` — Training loss over steps (0.30 → 0.006)
- `figures/val_chamfer_distance.png` — Validation CD over epochs (best: 0.00930 at epoch 40)
- `figures/results_comparison.png` — Bar chart comparing old vs new model metrics

### Synthetic Test Results (each is input image + predicted blue + ground truth green + side view)
- `figures/synth_airplane.png` — Airplane reconstruction
- `figures/synth_car1.png` — Car reconstruction
- `figures/synth_lamp.png` — Lamp reconstruction
- `figures/synth_cabinet.png` — Cabinet reconstruction

### Real-World Inference Results (each is original photo + background removed + no TTA + with TTA)
- `figures/real_car.png` — BMW car from internet photo
- `figures/real_monitor.png` — Computer monitor
- `figures/real_lamp.png` — Desk lamp
- `figures/real_rifle2.png` — Rifle

### TTA Before/After
- `figures/real_airplane.png` — Airplane: blue = no TTA (scattered), orange = with TTA (tighter)
- `figures/real_rifle1.png` — Rifle: blue = no TTA, orange = with TTA

### Chair Comparison (old model vs new model)
- `figures/old_chair1.png` — Old 1024-pt model (100 epochs) — better chair structure
- `figures/limitation_chair1.png` — New 2048-pt model (40 epochs) — denser but scattered

## Image Dimensions
- Synthetic images: 2250×600px (aspect ~3.75:1) — wide panoramic strips
- Real-world images: 2400×600px (aspect 4:1) — wide panoramic strips
- Training curves: 1800×900px and 1500×900px
- Results comparison: 2100×900px

## Where to Add Each Figure in main.tex

### Figure 1: Architecture Diagram
- Already exists as a text box in Section 3.1
- Keep as-is (no image needed)

### Figure 2: Training Curves (after Section 3.2 Training, after the Data Augmentation paragraph)
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

### Figure 3: Synthetic Results (at the start of Section 4.5 Qualitative Results)
Use `figure*` for full-width across both columns:
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

### Figure 4: Real-World Results (after Figure 3 in Section 4.5)
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
\caption{Real-world inference on internet photographs. From left to right: car, monitor, desk lamp, rifle. Background removal + TTA produces recognizable 3D shapes.}
\label{fig:real_results}
\end{figure*}
```

### Figure 5: TTA Effect (in Section 4.3 Domain Adaptation Analysis, after the TTA paragraph)
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

### Figure 6: Chair Comparison (in Section 4.5 after the qualitative text, or in Section 5 Discussion)
```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.48\linewidth]{figures/old_chair1.png}
\hfill
\includegraphics[width=0.48\linewidth]{figures/limitation_chair1.png}
\caption{Chair comparison. \textbf{Left:} Old model (1024 pts, 100 epochs) captures legs and structure. \textbf{Right:} New model (2048 pts, 40 epochs) is denser but scattered on thin structures.}
\label{fig:chair_comparison}
\end{figure}
```

### Figure 7: Results Comparison Bar Chart (in Section 4.2 next to or after Table 2)
```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.95\linewidth]{figures/results_comparison.png}
\caption{Visual comparison of old (1024-pt, 100 epochs) vs new (2048-pt, 40 epochs) model across Chamfer Distance and F-Score metrics.}
\label{fig:results_bar}
\end{figure}
```

## Text References to Add

After adding figures, make sure the text references them. Add these references where appropriate:

- In training section: "Training curves are shown in Figure~\ref{fig:training_curves}."
- In main results: "A visual comparison is shown in Figure~\ref{fig:results_bar}."
- In TTA analysis: "The effect of TTA is visualized in Figure~\ref{fig:tta_effect}."
- In qualitative results: "Figure~\ref{fig:synth_results}" and "Figure~\ref{fig:real_results}" and "Figure~\ref{fig:chair_comparison}"

## Page Limit Warning

The report MUST be 4-8 pages including everything (figures, references, appendices). With all these figures it may push to 7-8 pages. If it goes over 8 pages:
1. First try: use `\begin{figure}[t]` instead of `figure*` for the synthetic/real results (single column instead of full width)
2. Second try: remove Figure 7 (results bar chart) since Table 2 already has the same data
3. Third try: combine TTA figure and chair comparison into one figure

## Compilation

Compile with:
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
