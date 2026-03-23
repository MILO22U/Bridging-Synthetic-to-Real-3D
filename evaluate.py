"""
Evaluation Script — Measures domain gap and runs all adaptation strategies.

Usage:
    python evaluate.py --checkpoint checkpoints/base_pretrained/best_model.pt
    python evaluate.py --checkpoint checkpoints/base_pretrained/best_model.pt --tta
    python evaluate.py --checkpoint checkpoints/base_pretrained/best_model.pt --adain
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

# ── FIX #1: flat sibling imports (matches train.py style) ──
from config import get_config, config_from_yaml
from model import build_model
from losses import chamfer_distance, f_score
from dataset import (
    GSODataset, get_val_transform, create_dataloaders,
)
from strategies import TestTimeAugmentation


@torch.no_grad()
def evaluate_dataset(model, dataloader, device, name="test", use_tta=False, tta=None):
    """Evaluate model on a dataset, return per-sample metrics."""
    model.eval()
    
    results = defaultdict(list)
    all_cd = []
    all_fs = {0.01: [], 0.02: [], 0.05: []}
    
    for batch in tqdm(dataloader, desc=f"Evaluating {name}"):
        # Handle both tuple returns (dataset.py) and dict returns
        if isinstance(batch, (list, tuple)):
            images, gt_points = batch[0], batch[1]
        else:
            images, gt_points = batch["image"], batch["points"]
        
        images = images.to(device)
        gt_points = gt_points.to(device)
        
        # Predict
        if use_tta and tta is not None:
            pred_points = tta(model, images)
        else:
            pred_points = model(images)
        
        # Per-sample metrics
        B = images.shape[0]
        for i in range(B):
            pred_i = pred_points[i:i+1]
            gt_i = gt_points[i:i+1]
            
            cd, _, _ = chamfer_distance(pred_i, gt_i)
            all_cd.append(cd.item())
            
            for thresh in [0.01, 0.02, 0.05]:
                fs = f_score(pred_i, gt_i, threshold=thresh)
                all_fs[thresh].append(fs.item())
    
    # Aggregate
    metrics = {
        "chamfer_distance": {
            "mean": float(np.mean(all_cd)) if all_cd else 0.0,
            "std": float(np.std(all_cd)) if all_cd else 0.0,
            "median": float(np.median(all_cd)) if all_cd else 0.0,
            "n": len(all_cd),
        },
    }
    for thresh in [0.01, 0.02, 0.05]:
        metrics[f"fscore_{thresh}"] = {
            "mean": float(np.mean(all_fs[thresh])) if all_fs[thresh] else 0.0,
            "std": float(np.std(all_fs[thresh])) if all_fs[thresh] else 0.0,
        }
    
    return metrics


def print_metrics(metrics: dict, name: str):
    """Pretty print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"Results: {name}")
    print(f"{'='*60}")
    
    cd = metrics["chamfer_distance"]
    print(f"  Chamfer Distance: {cd['mean']:.6f} +/- {cd['std']:.6f} "
          f"(median: {cd['median']:.6f}, n={cd['n']})")
    
    for thresh in [0.01, 0.02, 0.05]:
        key = f"fscore_{thresh}"
        if key in metrics:
            fs = metrics[key]
            print(f"  F-Score@{thresh}: {fs['mean']:.4f} +/- {fs['std']:.4f}")


def domain_gap_report(syn_metrics: dict, real_metrics: dict):
    """Generate domain gap analysis report."""
    syn_cd = syn_metrics["chamfer_distance"]["mean"]
    real_cd = real_metrics["chamfer_distance"]["mean"]
    gap = real_cd - syn_cd
    ratio = real_cd / max(syn_cd, 1e-8)
    
    print(f"\n{'='*60}")
    print(f"DOMAIN GAP ANALYSIS")
    print(f"{'='*60}")
    print(f"  Synthetic CD:     {syn_cd:.6f}")
    print(f"  Real-World CD:    {real_cd:.6f}")
    print(f"  Absolute Gap:     {gap:.6f}")
    print(f"  Relative Gap:     {ratio:.2f}x")
    print(f"  Gap Percentage:   {(gap / max(syn_cd, 1e-8)) * 100:.1f}%")
    
    for thresh in [0.01, 0.02, 0.05]:
        syn_fs = syn_metrics.get(f"fscore_{thresh}", {}).get("mean", 0)
        real_fs = real_metrics.get(f"fscore_{thresh}", {}).get("mean", 0)
        print(f"  F-Score@{thresh} — Syn: {syn_fs:.4f}, Real: {real_fs:.4f}, "
              f"Drop: {syn_fs - real_fs:.4f}")
    
    return {
        "synthetic_cd": syn_cd,
        "real_cd": real_cd,
        "absolute_gap": gap,
        "relative_gap": ratio,
        "gap_percentage": (gap / max(syn_cd, 1e-8)) * 100,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3D Reconstruction Model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--experiment", type=str, default="base",
                       help="Experiment config to use")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config.yaml (overrides --experiment)")
    parser.add_argument("--tta", action="store_true",
                       help="Use Test-Time Augmentation (Strategy 2)")
    parser.add_argument("--tta_n", type=int, default=10,
                       help="Number of TTA augmentations")
    parser.add_argument("--output", type=str, default="./eval_results",
                       help="Output directory for results")
    args = parser.parse_args()
    
    # ── FIX #2: config loading matches train.py ─────────────
    if args.config:
        cfg = config_from_yaml(args.config)
    else:
        cfg = get_config(args.experiment)
    
    cfg_dict = cfg.to_dict()   # model.py and dataset.py need dict format
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    # ── FIX #3: build_model expects dict, returns (model, disc) ─
    print(f"Loading model from {args.checkpoint}")
    model, _ = build_model(cfg_dict)
    model = model.to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # ── FIX #4: create_dataloaders expects dict, returns tuple ──
    _, _, test_loader = create_dataloaders(cfg_dict)
    
    # GSO real-world loader
    gso_transform = get_val_transform(cfg_dict['augmentation'])
    gso_dataset = GSODataset(cfg.data.gso_root, num_points=cfg.data.num_points,
                             transform=gso_transform)
    gso_loader = None
    if len(gso_dataset) > 0:
        from torch.utils.data import DataLoader
        gso_loader = DataLoader(gso_dataset, batch_size=cfg.training.batch_size,
                                shuffle=False, num_workers=cfg.data.num_workers)
    
    # TTA setup
    tta = None
    if args.tta:
        tta = TestTimeAugmentation(num_augments=args.tta_n)
        print(f"Using Test-Time Augmentation with {args.tta_n} augments")
    
    all_results = {}
    syn_metrics = None
    
    # Evaluate synthetic test set
    if len(test_loader) > 0:
        suffix = "_tta" if args.tta else ""
        syn_metrics = evaluate_dataset(
            model, test_loader, device,
            name=f"Synthetic Test{suffix}",
            use_tta=args.tta, tta=tta
        )
        print_metrics(syn_metrics, f"Synthetic Test Set{suffix}")
        all_results[f"synthetic{suffix}"] = syn_metrics
    
    # Evaluate real-world test set
    if gso_loader is not None:
        suffix = "_tta" if args.tta else ""
        real_metrics = evaluate_dataset(
            model, gso_loader, device,
            name=f"Real-World Test{suffix}",
            use_tta=args.tta, tta=tta
        )
        print_metrics(real_metrics, f"Real-World Test Set (GSO){suffix}")
        all_results[f"real{suffix}"] = real_metrics
        
        # Domain gap report
        if syn_metrics is not None:
            gap_info = domain_gap_report(syn_metrics, real_metrics)
            all_results["domain_gap"] = gap_info
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(args.output, f"results_{cfg.experiment_name}.json")
    
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return obj
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
