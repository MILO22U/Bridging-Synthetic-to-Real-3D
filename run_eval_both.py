"""Evaluate both old 1024-pt and new 2048-pt models on synthetic test."""
import torch
import yaml
import numpy as np
from tqdm import tqdm
from model import HybridReconstructor
from dataset import create_dataloaders
from losses import chamfer_distance, f_score


def eval_model(model, test_loader, device, label):
    all_cd = []
    all_fs = {0.01: [], 0.02: [], 0.05: []}
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Eval {label}"):
            images, gt_points = batch[0].to(device), batch[1].to(device)
            pred = model(images)
            for i in range(images.shape[0]):
                cd, _, _ = chamfer_distance(pred[i:i+1], gt_points[i:i+1])
                all_cd.append(cd.item())
                for t in [0.01, 0.02, 0.05]:
                    fs = f_score(pred[i:i+1], gt_points[i:i+1], threshold=t)
                    all_fs[t].append(fs.item())

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Chamfer Distance: {np.mean(all_cd):.6f} +/- {np.std(all_cd):.6f}")
    print(f"  Median CD:        {np.median(all_cd):.6f}")
    print(f"  F-Score@0.01:     {np.mean(all_fs[0.01]):.4f}")
    print(f"  F-Score@0.02:     {np.mean(all_fs[0.02]):.4f}")
    print(f"  F-Score@0.05:     {np.mean(all_fs[0.05]):.4f}")
    print(f"  N samples:        {len(all_cd)}")
    return {
        "cd_mean": np.mean(all_cd), "cd_std": np.std(all_cd),
        "cd_median": np.median(all_cd),
        "fs01": np.mean(all_fs[0.01]), "fs02": np.mean(all_fs[0.02]),
        "fs05": np.mean(all_fs[0.05]), "n": len(all_cd),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    # ── Old model: 1024 points ──
    cfg1 = {**cfg}
    cfg1["model"] = {**cfg["model"], "num_query_tokens": 1024}
    cfg1["data"] = {**cfg["data"], "num_points": 1024}

    model1 = HybridReconstructor(cfg1["model"]).to(device)
    ckpt1 = torch.load("checkpoints/base_pretrained/best_model.pt",
                        map_location=device, weights_only=False)
    model1.load_state_dict(ckpt1["model_state_dict"])
    model1.eval()
    print(f"Old model loaded: epoch {ckpt1['epoch']}, best_val_cd={ckpt1.get('best_val_cd', 'N/A')}")

    _, _, test_loader1 = create_dataloaders(cfg1)
    print(f"Test loader (1024 pts): {len(test_loader1)} batches")
    r1 = eval_model(model1, test_loader1, device, "OLD MODEL (1024 pts)")
    del model1
    torch.cuda.empty_cache()

    # ── New model: 2048 points ──
    import os
    ckpt_path = "checkpoints/retrain_2048/best.pt"
    if not os.path.exists(ckpt_path):
        print(f"\n{ckpt_path} not found, skipping new model eval.")
        return

    cfg2 = {**cfg}
    cfg2["model"] = {**cfg["model"], "num_query_tokens": 2048}
    cfg2["data"] = {**cfg["data"], "num_points": 2048}

    model2 = HybridReconstructor(cfg2["model"]).to(device)
    ckpt2 = torch.load(ckpt_path, map_location=device, weights_only=False)
    model2.load_state_dict(ckpt2["model_state_dict"])
    model2.eval()
    print(f"\nNew model loaded: epoch {ckpt2['epoch']}, best_val_cd={ckpt2.get('best_val_cd', 'N/A')}")

    _, _, test_loader2 = create_dataloaders(cfg2)
    print(f"Test loader (2048 pts): {len(test_loader2)} batches")
    r2 = eval_model(model2, test_loader2, device, "NEW MODEL (2048 pts)")

    # ── Comparison ──
    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'Old (1024)':<15} {'New (2048)':<15}")
    print(f"  {'CD Mean':<20} {r1['cd_mean']:<15.6f} {r2['cd_mean']:<15.6f}")
    print(f"  {'CD Median':<20} {r1['cd_median']:<15.6f} {r2['cd_median']:<15.6f}")
    print(f"  {'F@0.01':<20} {r1['fs01']:<15.4f} {r2['fs01']:<15.4f}")
    print(f"  {'F@0.02':<20} {r1['fs02']:<15.4f} {r2['fs02']:<15.4f}")
    print(f"  {'F@0.05':<20} {r1['fs05']:<15.4f} {r2['fs05']:<15.4f}")


if __name__ == "__main__":
    main()
