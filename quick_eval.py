"""Quick evaluation: synthetic test + real photos."""
import torch
import yaml
import os
import numpy as np
from model import HybridReconstructor
from dataset import create_dataloaders
from losses import chamfer_distance, f_score

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)

    # Load model — checkpoint used 1024 query tokens
    cfg['model']['num_query_tokens'] = 1024
    cfg['data']['num_points'] = 1024
    model = HybridReconstructor(cfg['model']).to(device)
    ckpt = torch.load('checkpoints/base_pretrained/best_model.pt',
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded epoch {ckpt['epoch']} (val_cd={ckpt.get('val_cd', '?')})")

    # Create test loader
    _, _, test_loader = create_dataloaders(cfg)
    print(f"Test samples: {len(test_loader.dataset)}")

    # Evaluate
    all_cd = []
    all_fs = {0.01: [], 0.02: [], 0.05: []}

    with torch.no_grad():
        for i, (images, gt_points, _) in enumerate(test_loader):
            images = images.to(device)
            gt_points = gt_points.to(device)

            pred = model(images)

            cd_loss, _, _ = chamfer_distance(pred, gt_points, bidirectional=True)
            all_cd.append(cd_loss.item())

            for t in [0.01, 0.02, 0.05]:
                fs = f_score(pred, gt_points, threshold=t)
                all_fs[t].append(fs.mean().item())

            if (i + 1) % 50 == 0:
                print(f"  Batch {i+1}/{len(test_loader)}")

    print("\n=== Synthetic Test Results ===")
    print(f"  Chamfer Distance: {np.mean(all_cd):.6f}")
    for t in [0.01, 0.02, 0.05]:
        print(f"  F-Score@{t}: {np.mean(all_fs[t]):.4f}")

    # Check real photos
    real_dir = cfg['data'].get('real_images_dir', './data/real_photos')
    if os.path.exists(real_dir):
        photos = [f for f in os.listdir(real_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"\nReal photos available: {len(photos)} in {real_dir}")
    else:
        print(f"\nNo real photos at {real_dir}")

if __name__ == '__main__':
    main()
