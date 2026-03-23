"""
Generate training loss curves and evaluation metric plots.

Reads from TensorBoard logs or reconstructs from checkpoint history.

Usage: python plot_training_curves.py
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Try reading TensorBoard logs
def read_tensorboard_logs(log_dir):
    """Try to read TensorBoard event files."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(log_dir)
        ea.Reload()
        
        data = {}
        for tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            data[tag] = {
                'steps': [e.step for e in events],
                'values': [e.value for e in events],
            }
        return data
    except Exception as e:
        print(f"Could not read TensorBoard logs: {e}")
        return None


def plot_from_tensorboard(log_dir, output_dir):
    """Generate plots from TensorBoard logs."""
    data = read_tensorboard_logs(log_dir)
    if data is None:
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Available tags: {list(data.keys())}")
    
    # Plot 1: Training loss curve
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'train/recon_loss' in data:
        d = data['train/recon_loss']
        ax.plot(d['steps'], d['values'], alpha=0.3, color='blue', label='Per-batch loss')
        # Smoothed version
        window = min(50, len(d['values']) // 10 + 1)
        if window > 1:
            smoothed = np.convolve(d['values'], np.ones(window)/window, mode='valid')
            ax.plot(d['steps'][window-1:], smoothed, color='blue', linewidth=2, label='Smoothed')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Chamfer Distance Loss', fontsize=12)
    ax.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_loss_curve.png'), dpi=150)
    plt.close()
    print(f"  Saved: training_loss_curve.png")
    
    # Plot 2: Validation CD
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'val/chamfer_distance' in data:
        d = data['val/chamfer_distance']
        ax.plot(d['steps'], d['values'], 'o-', color='red', markersize=4, linewidth=2)
        best_idx = np.argmin(d['values'])
        ax.axhline(y=d['values'][best_idx], color='green', linestyle='--', alpha=0.5,
                   label=f"Best: {d['values'][best_idx]:.4f} (epoch {d['steps'][best_idx]})")
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Chamfer Distance', fontsize=12)
    ax.set_title('Validation Chamfer Distance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'val_chamfer_distance.png'), dpi=150)
    plt.close()
    print(f"  Saved: val_chamfer_distance.png")
    
    # Plot 3: F-Scores
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'0.01': 'red', '0.02': 'orange', '0.05': 'green'}
    for thresh in ['0.01', '0.02', '0.05']:
        tag = f'val/fscore_{thresh}'
        if tag in data:
            d = data[tag]
            ax.plot(d['steps'], d['values'], 'o-', markersize=3, linewidth=2,
                   color=colors[thresh], label=f'F@{thresh}')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F-Score', fontsize=12)
    ax.set_title('Validation F-Scores', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'val_fscores.png'), dpi=150)
    plt.close()
    print(f"  Saved: val_fscores.png")
    
    # Plot 4: Learning rate
    fig, ax = plt.subplots(figsize=(10, 6))
    for tag_name, label, color in [
        ('train/lr_encoder', 'Encoder LR', 'blue'),
        ('train/lr_decoder', 'Decoder LR', 'red'),
    ]:
        if tag_name in data:
            d = data[tag_name]
            ax.plot(d['steps'], d['values'], color=color, linewidth=2, label=label)
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_rate.png'), dpi=150)
    plt.close()
    print(f"  Saved: learning_rate.png")
    
    # Plot 5: Summary bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Base vs TTA comparison
    base_results = {'CD': 0.01541, 'F@0.01': 0.0152, 'F@0.02': 0.0829, 'F@0.05': 0.4387}
    tta_results = {'CD': 0.01531, 'F@0.01': 0.0150, 'F@0.02': 0.0834, 'F@0.05': 0.4400}
    
    # CD comparison
    methods = ['Base Model', 'With TTA']
    cd_vals = [base_results['CD'], tta_results['CD']]
    bars = axes[0].bar(methods, cd_vals, color=['steelblue', 'coral'], width=0.5)
    axes[0].set_ylabel('Chamfer Distance (lower is better)', fontsize=11)
    axes[0].set_title('Chamfer Distance Comparison', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, cd_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.0002,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # F-Score comparison
    x = np.arange(3)
    width = 0.35
    f_base = [base_results['F@0.01'], base_results['F@0.02'], base_results['F@0.05']]
    f_tta = [tta_results['F@0.01'], tta_results['F@0.02'], tta_results['F@0.05']]
    bars1 = axes[1].bar(x - width/2, f_base, width, label='Base', color='steelblue')
    bars2 = axes[1].bar(x + width/2, f_tta, width, label='TTA', color='coral')
    axes[1].set_ylabel('F-Score (higher is better)', fontsize=11)
    axes[1].set_title('F-Score Comparison', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['F@0.01', 'F@0.02', 'F@0.05'])
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'results_comparison.png'), dpi=150)
    plt.close()
    print(f"  Saved: results_comparison.png")
    
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="./outputs/logs/base_pretrained")
    parser.add_argument("--output", default="./visualizations/plots")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Reading logs from: {args.log_dir}")
    print(f"Saving plots to: {args.output}")
    print()
    
    success = plot_from_tensorboard(args.log_dir, args.output)
    
    if not success:
        print("TensorBoard reading failed. Install tensorboard:")
        print("  pip install tensorboard")
        print("Then re-run this script.")
    else:
        print(f"\nAll plots saved to {args.output}/")


if __name__ == "__main__":
    main()
