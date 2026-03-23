"""Compare synthetic predictions (which work) vs real predictions (which don't)."""
import torch
import yaml
import numpy as np
from model import HybridReconstructor
from dataset import create_dataloaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

cfg['model']['num_query_tokens'] = 1024
cfg['data']['num_points'] = 1024
model = HybridReconstructor(cfg['model']).to(device)
ckpt = torch.load('checkpoints/base_pretrained/best_model.pt',
                   map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

_, _, test_loader = create_dataloaders(cfg)

# Get a few synthetic predictions
with torch.no_grad():
    for images, gt_points, meta in test_loader:
        images = images.to(device)
        gt_points = gt_points.to(device)
        pred = model(images)

        for i in range(min(3, images.shape[0])):
            p = pred[i].cpu().numpy()
            g = gt_points[i].cpu().numpy()
            print(f"Synthetic sample {i}:")
            print(f"  Pred  spread={p.std(axis=0).mean():.4f} extent={p.max(axis=0)-p.min(axis=0)}")
            print(f"  GT    spread={g.std(axis=0).mean():.4f} extent={g.max(axis=0)-g.min(axis=0)}")
        break
