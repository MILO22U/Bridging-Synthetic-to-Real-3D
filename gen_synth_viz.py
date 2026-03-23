"""Generate synthetic test visualizations: image + predicted + GT point clouds."""
import torch
import yaml
import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import HybridReconstructor
from dataset import create_dataloaders

def plot_pc(ax, points, color='steelblue', title='', elev=25, azim=45):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=color, s=0.3, alpha=0.5)
    ax.set_title(title, fontsize=8)
    r = max(np.max(np.abs(points)) * 1.1, 0.5)
    ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)
    ax.view_init(elev=elev, azim=azim)

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)

    cfg['model']['num_query_tokens'] = 2048
    cfg['data']['num_points'] = 2048
    model = HybridReconstructor(cfg['model']).to(device)
    ckpt = torch.load('checkpoints/retrain_2048/best.pt',
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    _, _, test_loader = create_dataloaders(cfg)
    save_dir = './visualizations/synthetic_test_2048'
    os.makedirs(save_dir, exist_ok=True)

    # Get 20 diverse samples
    count = 0
    max_samples = 20
    with torch.no_grad():
        for images, gt_points, model_ids in test_loader:
            images = images.to(device)
            gt_points = gt_points.to(device)
            pred = model(images)

            for i in range(images.shape[0]):
                if count >= max_samples:
                    break

                fig = plt.figure(figsize=(15, 4))

                # Input image
                ax1 = fig.add_subplot(141)
                img_vis = denormalize(images[i]).permute(1, 2, 0).numpy()
                ax1.imshow(img_vis)
                ax1.set_title('Input Image', fontsize=9)
                ax1.axis('off')

                # Predicted point cloud - view 1
                ax2 = fig.add_subplot(142, projection='3d')
                plot_pc(ax2, pred[i], 'steelblue', 'Predicted', elev=25, azim=45)

                # GT point cloud - view 1
                ax3 = fig.add_subplot(143, projection='3d')
                plot_pc(ax3, gt_points[i], 'forestgreen', 'Ground Truth', elev=25, azim=45)

                # Predicted - view 2
                ax4 = fig.add_subplot(144, projection='3d')
                plot_pc(ax4, pred[i], 'coral', 'Predicted (side)', elev=10, azim=135)

                plt.suptitle(f'{model_ids[i]}', fontsize=9)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'synth_{count:02d}.png'), dpi=150)
                plt.close()
                count += 1

            if count >= max_samples:
                break

    print(f"Saved {count} visualizations to {save_dir}/")

if __name__ == '__main__':
    main()
