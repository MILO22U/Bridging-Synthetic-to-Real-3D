"""Generate side-by-side comparison: old 1024-pt vs new 2048-pt model on synthetic test."""
import torch
import yaml
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import HybridReconstructor
from dataset import create_dataloaders
from losses import chamfer_distance


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

    # Load old model (1024 pts)
    cfg1 = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()}
    cfg1['model'] = {**cfg['model'], 'num_query_tokens': 1024}
    cfg1['data'] = {**cfg['data'], 'num_points': 1024}
    model_old = HybridReconstructor(cfg1['model']).to(device)
    ckpt_old = torch.load('checkpoints/base_pretrained/best_model.pt',
                           map_location=device, weights_only=False)
    model_old.load_state_dict(ckpt_old['model_state_dict'])
    model_old.eval()
    print(f"Old model: epoch {ckpt_old['epoch']}")

    # Load new model (2048 pts)
    cfg2 = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()}
    cfg2['model'] = {**cfg['model'], 'num_query_tokens': 2048}
    cfg2['data'] = {**cfg['data'], 'num_points': 2048}
    model_new = HybridReconstructor(cfg2['model']).to(device)
    ckpt_new = torch.load('checkpoints/retrain_2048/best.pt',
                           map_location=device, weights_only=False)
    model_new.load_state_dict(ckpt_new['model_state_dict'])
    model_new.eval()
    print(f"New model: epoch {ckpt_new['epoch']}")

    # Use 1024-pt dataloader (both models can predict, GT will be 1024 for fair CD comparison)
    _, _, test_loader = create_dataloaders(cfg1)

    save_dir = './visualizations/comparison_old_vs_new'
    os.makedirs(save_dir, exist_ok=True)

    count = 0
    max_samples = 15
    with torch.no_grad():
        for batch in test_loader:
            images, gt_points = batch[0].to(device), batch[1].to(device)
            model_ids = batch[2] if len(batch) > 2 else [f"sample_{count+i}" for i in range(images.shape[0])]

            pred_old = model_old(images)
            pred_new = model_new(images)

            for i in range(images.shape[0]):
                if count >= max_samples:
                    break

                # Compute CD for each
                cd_old, _, _ = chamfer_distance(pred_old[i:i+1], gt_points[i:i+1])
                cd_new_vs_1024, _, _ = chamfer_distance(pred_new[i:i+1], gt_points[i:i+1])

                fig = plt.figure(figsize=(20, 4))

                # Input image
                ax1 = fig.add_subplot(151)
                img_vis = denormalize(images[i]).permute(1, 2, 0).numpy()
                ax1.imshow(img_vis)
                ax1.set_title('Input Image', fontsize=9)
                ax1.axis('off')

                # Old prediction
                ax2 = fig.add_subplot(152, projection='3d')
                plot_pc(ax2, pred_old[i], 'steelblue',
                        f'Old (1024pts)\nCD={cd_old.item():.5f}')

                # New prediction
                ax3 = fig.add_subplot(153, projection='3d')
                plot_pc(ax3, pred_new[i], 'coral',
                        f'New (2048pts)\nCD={cd_new_vs_1024.item():.5f}')

                # GT
                ax4 = fig.add_subplot(154, projection='3d')
                plot_pc(ax4, gt_points[i], 'forestgreen', 'Ground Truth')

                # New prediction side view
                ax5 = fig.add_subplot(155, projection='3d')
                plot_pc(ax5, pred_new[i], 'coral', 'New (side view)',
                        elev=10, azim=135)

                mid = model_ids[i] if isinstance(model_ids[i], str) else str(model_ids[i])
                plt.suptitle(f'{mid}', fontsize=9)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'cmp_{count:02d}.png'), dpi=150)
                plt.close()
                count += 1

            if count >= max_samples:
                break

    print(f"Saved {count} comparison visualizations to {save_dir}/")


if __name__ == '__main__':
    main()
