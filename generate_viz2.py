"""
Generate visualizations2/ — synthetic test samples + old vs new comparison charts.

Usage:
    python generate_viz2.py --checkpoint checkpoints/retrain_2048/best.pt --config config_2048.yaml
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import config_from_yaml
from model import build_model
from dataset import create_dataloaders
from losses import chamfer_distance, f_score
from visualize import plot_comparison


@torch.no_grad()
def generate_synthetic_viz(model, dataloader, device, save_dir, n_samples=30):
    """Generate image | pred | gt comparisons for synthetic test samples."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    count = 0
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            images, gt_points = batch[0], batch[1]
        else:
            images, gt_points = batch["image"], batch["points"]

        images = images.to(device)
        gt_points = gt_points.to(device)
        pred_points = model(images)

        for i in range(images.shape[0]):
            if count >= n_samples:
                return count

            pred_i = pred_points[i:i+1]
            gt_i = gt_points[i:i+1]
            cd, _, _ = chamfer_distance(pred_i, gt_i)
            fs = f_score(pred_i, gt_i, threshold=0.02)

            save_path = os.path.join(save_dir, f'sample_{count:03d}_cd{cd.item():.5f}.png')
            plot_comparison(
                images[i], pred_points[i], gt_points[i],
                title=f'Sample {count} | CD: {cd.item():.5f} | F@0.02: {fs.item():.3f}',
                save_path=save_path,
            )
            count += 1
            if count % 5 == 0:
                print(f"  Saved {count}/{n_samples} visualizations")

    return count


def generate_comparison_charts(save_dir):
    """Generate comparison bar charts between old (1024-pt) and new (2048-pt) models."""
    os.makedirs(save_dir, exist_ok=True)

    # Old model results (1024-pt, base_pretrained, with TTA)
    old_results = {
        "name": "1024-pt (100ep, TTA)",
        "cd": 0.005821,
        "f01": 0.0223,
        "f02": 0.1428,
        "f05": 0.6822,
    }

    # New model results (2048-pt, retrain_2048, no TTA)
    new_no_tta = {
        "name": "2048-pt (100ep, no TTA)",
        "cd": 0.008560,
        "f01": 0.0448,
        "f02": 0.2068,
        "f05": 0.6858,
    }

    # Check if TTA results exist
    tta_path = os.path.join(save_dir, '..', 'eval_results_tta', 'results_retrain_2048.json')
    new_tta = None
    if os.path.exists(tta_path):
        with open(tta_path) as f:
            tta_data = json.load(f)
        if 'synthetic_tta' in tta_data:
            s = tta_data['synthetic_tta']
            new_tta = {
                "name": "2048-pt (100ep, TTA)",
                "cd": s['chamfer_distance']['mean'],
                "f01": s['fscore_0.01']['mean'],
                "f02": s['fscore_0.02']['mean'],
                "f05": s['fscore_0.05']['mean'],
            }

    models = [old_results, new_no_tta]
    if new_tta:
        models.append(new_tta)

    names = [m["name"] for m in models]
    colors = ['#4CAF50', '#2196F3', '#FF9800'][:len(models)]

    # ── Chart 1: Chamfer Distance ──
    fig, ax = plt.subplots(figsize=(8, 5))
    cds = [m["cd"] for m in models]
    bars = ax.bar(names, cds, color=colors, width=0.5)
    for bar, val in zip(bars, cds):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.0002,
                f'{val:.5f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Chamfer Distance (lower is better)')
    ax.set_title('Chamfer Distance Comparison: Old vs New Model')
    ax.tick_params(axis='x', rotation=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_chamfer_distance.png'), dpi=150)
    plt.close()
    print("  Saved comparison_chamfer_distance.png")

    # ── Chart 2: F-Scores ──
    fig, ax = plt.subplots(figsize=(10, 5))
    thresholds = ['F@0.01', 'F@0.02', 'F@0.05']
    x = np.arange(len(thresholds))
    width = 0.25

    for idx, m in enumerate(models):
        fscores = [m["f01"], m["f02"], m["f05"]]
        offset = (idx - (len(models)-1)/2) * width
        bars = ax.bar(x + offset, fscores, width, label=m["name"], color=colors[idx])
        for bar, val in zip(bars, fscores):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('F-Score (higher is better)')
    ax.set_title('F-Score Comparison: Old vs New Model')
    ax.set_xticks(x)
    ax.set_xticklabels(thresholds)
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_fscores.png'), dpi=150)
    plt.close()
    print("  Saved comparison_fscores.png")

    # ── Chart 3: Summary table as image ──
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    headers = ['Model', 'CD ↓', 'F@0.01 ↑', 'F@0.02 ↑', 'F@0.05 ↑']
    rows = []
    for m in models:
        rows.append([m["name"], f'{m["cd"]:.5f}', f'{m["f01"]:.4f}',
                      f'{m["f02"]:.4f}', f'{m["f05"]:.4f}'])

    table = ax.table(cellText=rows, colLabels=headers, loc='center',
                      cellLoc='center', colColours=['#E3F2FD']*5)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    ax.set_title('Model Comparison Summary', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_summary_table.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved comparison_summary_table.png")

    # Save comparison JSON
    summary = {m["name"]: {k: v for k, v in m.items() if k != "name"} for m in models}
    with open(os.path.join(save_dir, 'comparison_metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("  Saved comparison_metrics.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', default='./visualizations2')
    parser.add_argument('--n_samples', type=int, default=30)
    args = parser.parse_args()

    cfg = config_from_yaml(args.config)
    cfg_dict = cfg.to_dict()
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    # Load model
    print("Loading model...")
    model, _ = build_model(cfg_dict)
    model = model.to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Loaded epoch {ckpt['epoch']}, best CD: {ckpt.get('best_cd', 'N/A')}")

    # Data
    _, _, test_loader = create_dataloaders(cfg_dict)
    print(f"  Test samples: {len(test_loader.dataset)}")

    # Generate synthetic test visualizations
    print(f"\n{'='*60}")
    print(f"Generating synthetic test visualizations...")
    print(f"{'='*60}")
    syn_dir = os.path.join(args.output, 'synthetic_test')
    n = generate_synthetic_viz(model, test_loader, device, syn_dir, args.n_samples)
    print(f"  Done: {n} visualizations saved to {syn_dir}/")

    # Generate comparison charts
    print(f"\n{'='*60}")
    print(f"Generating comparison charts...")
    print(f"{'='*60}")
    comp_dir = os.path.join(args.output, 'comparison')
    generate_comparison_charts(comp_dir)

    print(f"\n{'='*60}")
    print(f"All done! Output in {args.output}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
