"""
Visualization utilities for 3D point cloud reconstruction.

- Point cloud rendering (matplotlib 3D)
- Side-by-side comparison (input image | predicted | ground truth)
- Domain gap bar charts
- Training curves
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch


def plot_point_cloud(points, title='', color='steelblue', figsize=(6, 6),
                     elev=30, azim=45, save_path=None):
    """
    Plot a single 3D point cloud.
    
    Args:
        points: (N, 3) numpy array or torch tensor
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=color, s=0.5, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    
    # Equal aspect ratio
    max_range = np.max(np.abs(points)) * 1.1
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def plot_comparison(image, pred_points, gt_points=None, title='',
                    save_path=None):
    """
    Side-by-side: input image | predicted point cloud | ground truth.
    
    Args:
        image: (3, H, W) tensor or (H, W, 3) numpy array
        pred_points: (N, 3)
        gt_points: (M, 3) or None
    """
    ncols = 3 if gt_points is not None else 2
    fig = plt.figure(figsize=(6 * ncols, 5))
    
    # Input image
    ax1 = fig.add_subplot(1, ncols, 1)
    if isinstance(image, torch.Tensor):
        img = image.cpu()
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img = img * std + mean
        img = img.clamp(0, 1).numpy()
    else:
        img = image
    ax1.imshow(img)
    ax1.set_title('Input Image')
    ax1.axis('off')
    
    # Predicted point cloud
    if isinstance(pred_points, torch.Tensor):
        pred_points = pred_points.cpu().numpy()
    ax2 = fig.add_subplot(1, ncols, 2, projection='3d')
    ax2.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2],
                c='steelblue', s=0.5, alpha=0.6)
    ax2.set_title('Predicted')
    ax2.view_init(30, 45)
    
    # Ground truth
    if gt_points is not None:
        if isinstance(gt_points, torch.Tensor):
            gt_points = gt_points.cpu().numpy()
        ax3 = fig.add_subplot(1, ncols, 3, projection='3d')
        ax3.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2],
                    c='coral', s=0.5, alpha=0.6)
        ax3.set_title('Ground Truth')
        ax3.view_init(30, 45)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def plot_domain_gap_bar(synth_metrics, real_metrics, thresholds,
                        title='Synthetic vs Real-World Performance',
                        save_path=None):
    """Bar chart comparing synthetic vs real metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Chamfer Distance
    ax = axes[0]
    labels = ['Synthetic', 'Real-World']
    values = [synth_metrics['chamfer_distance'], real_metrics['chamfer_distance']]
    colors = ['#4CAF50', '#FF5722']
    bars = ax.bar(labels, values, color=colors, width=0.5)
    ax.set_ylabel('Chamfer Distance (↓ better)')
    ax.set_title('Chamfer Distance Comparison')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11)
    
    # F-Scores
    ax = axes[1]
    x = np.arange(len(thresholds))
    width = 0.3
    synth_fs = [synth_metrics.get(f'f_score@{t}', 0) for t in thresholds]
    real_fs = [real_metrics.get(f'f_score@{t}', 0) for t in thresholds]
    
    ax.bar(x - width/2, synth_fs, width, label='Synthetic', color='#4CAF50')
    ax.bar(x + width/2, real_fs, width, label='Real-World', color='#FF5722')
    ax.set_ylabel('F-Score (↑ better)')
    ax.set_title('F-Score Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f'τ={t}' for t in thresholds])
    ax.legend()
    ax.set_ylim(0, 1.05)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def plot_adaptation_comparison(results_dict, metric='chamfer_distance',
                               title='Domain Adaptation Strategy Comparison',
                               save_path=None):
    """
    Bar chart comparing different adaptation strategies.
    
    Args:
        results_dict: {'baseline': metrics, 'augmentation': metrics, ...}
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    names = list(results_dict.keys())
    values = [results_dict[n].get(metric, 0) for n in names]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    bars = ax.bar(names, values, color=colors, width=0.6)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


def plot_training_curves(log_file, save_path=None):
    """Plot training and validation loss curves from log."""
    import json
    
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return
    
    with open(log_file) as f:
        logs = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(logs['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, logs['train_loss'], label='Train', color='steelblue')
    axes[0].plot(epochs, logs['val_loss'], label='Val', color='coral')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Chamfer Distance')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].set_yscale('log')
    
    # F-Score
    if 'val_fscore' in logs:
        axes[1].plot(epochs, logs['val_fscore'], label='Val F-Score', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('F-Score')
        axes[1].set_title('Validation F-Score')
        axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig


@torch.no_grad()
def generate_visualizations(model, dataloader, device, save_dir, n_samples=20):
    """Generate comparison visualizations for a set of samples."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    count = 0
    for batch in dataloader:
        if len(batch) == 3:
            images, gt_points, names = batch
        else:
            images, gt_points = batch[:2]
            names = [f'sample_{count + i}' for i in range(images.shape[0])]
        
        images = images.to(device)
        gt_points = gt_points.to(device)
        pred_points = model(images)
        
        for i in range(images.shape[0]):
            if count >= n_samples:
                return
            
            save_path = os.path.join(save_dir, f'{names[i]}.png')
            plot_comparison(
                images[i], pred_points[i], gt_points[i],
                title=f'{names[i]}', save_path=save_path,
            )
            count += 1
            print(f"  Saved visualization {count}/{n_samples}")
    
    print(f"Generated {count} visualizations in {save_dir}")
