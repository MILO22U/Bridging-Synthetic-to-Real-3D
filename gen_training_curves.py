"""Parse training_log.txt and generate training curves."""
import re
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_log(log_path):
    """Parse training log for per-epoch stats."""
    epochs = []
    train_cd = []
    val_cd = []
    best_cd = []

    with open(log_path, 'r') as f:
        lines = f.readlines()

    # Look for validation/epoch summary lines
    for line in lines:
        # Match epoch completion lines like "Epoch 5 | Train CD: 0.0123 | Val CD: 0.0098"
        m = re.search(r'Epoch\s+(\d+).*?Train.*?CD[:\s]+([0-9.]+).*?Val.*?CD[:\s]+([0-9.]+)', line)
        if m:
            epochs.append(int(m.group(1)))
            train_cd.append(float(m.group(2)))
            val_cd.append(float(m.group(3)))
            continue

        # Match "Val CD: X.XXX" or "val_cd: X.XXX"
        m = re.search(r'[Vv]al.*?CD[:\s]+([0-9.]+)', line)
        if m and 'best' not in line.lower():
            pass  # handled above

        # Match best model lines
        m = re.search(r'[Bb]est.*?CD[:\s]+([0-9.]+)', line)
        if m:
            best_cd.append(float(m.group(1)))

    # Also parse running CD from tqdm lines per epoch
    epoch_running_cd = {}
    for line in lines:
        m = re.search(r'Epoch\s+(\d+):\s+100%.*?CD[:,]\s*([0-9.]+)', line)
        if m:
            ep = int(m.group(1))
            cd = float(m.group(2))
            epoch_running_cd[ep] = cd

    return {
        'epochs': epochs, 'train_cd': train_cd, 'val_cd': val_cd,
        'best_cd': best_cd, 'epoch_running_cd': epoch_running_cd,
    }


def parse_epoch_summaries(log_path):
    """Parse epoch summary blocks from training log."""
    epochs_data = []
    with open(log_path, 'r') as f:
        content = f.read()

    # Look for patterns like:
    # Epoch X/40 summary  or  [Epoch X]  or  Epoch X:
    # followed by train loss, val CD, etc.
    blocks = re.split(r'(?=Epoch \d+[/:])', content)
    for block in blocks:
        m_epoch = re.match(r'Epoch\s+(\d+)', block)
        if not m_epoch:
            continue
        ep = int(m_epoch.group(1))

        # Find val CD
        m_val = re.search(r'[Vv]al.*?(?:CD|chamfer)[:\s=]+([0-9.]+)', block[:500])
        # Find train CD
        m_train = re.search(r'[Tt]rain.*?(?:CD|loss|chamfer)[:\s=]+([0-9.]+)', block[:500])

        if m_val:
            epochs_data.append({
                'epoch': ep,
                'val_cd': float(m_val.group(1)),
                'train_cd': float(m_train.group(1)) if m_train else None,
            })

    return epochs_data


def main():
    log_path = 'training_log.txt'
    save_dir = './visualizations/plots'
    os.makedirs(save_dir, exist_ok=True)

    data = parse_log(log_path)
    summaries = parse_epoch_summaries(log_path)

    # Use running CD from tqdm as training curve
    running = data['epoch_running_cd']

    if running:
        eps = sorted(running.keys())
        cds = [running[e] for e in eps]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(eps, cds, 'o-', color='blue', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Training Chamfer Distance', fontsize=12)
        ax.set_title('Training Loss Curve (2048-pt Retrain)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_loss_curve.png'), dpi=150)
        plt.close()
        print(f"Saved training_loss_curve.png ({len(eps)} epochs)")

    # Validation curve from summaries
    if summaries:
        eps_v = [s['epoch'] for s in summaries]
        val_cds = [s['val_cd'] for s in summaries]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(eps_v, val_cds, 'o-', color='red', linewidth=2, markersize=5)
        best_idx = np.argmin(val_cds)
        ax.axhline(y=val_cds[best_idx], color='green', linestyle='--', alpha=0.5,
                   label=f"Best: {val_cds[best_idx]:.6f} (epoch {eps_v[best_idx]})")
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Validation Chamfer Distance', fontsize=12)
        ax.set_title('Validation CD Over Training', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'val_chamfer_distance.png'), dpi=150)
        plt.close()
        print(f"Saved val_chamfer_distance.png ({len(eps_v)} points)")

    # Also parse raw tqdm lines for a smoother training curve
    if True:
        print("No epoch-completion lines found. Parsing tqdm progress...")
        steps = []
        cds = []
        with open(log_path) as f:
            for line in f:
                m = re.search(r'Epoch\s+(\d+):\s+\d+%\|.*?\|\s*(\d+)/(\d+).*?CD[:,]+\s*([0-9.]+)', line)
                if m:
                    ep = int(m.group(1))
                    batch = int(m.group(2))
                    total = int(m.group(3))
                    cd = float(m.group(4))
                    step = (ep - 1) * total + batch
                    steps.append(step)
                    cds.append(cd)

        if steps:
            # Subsample for plotting
            n = len(steps)
            idx = np.linspace(0, n-1, min(2000, n), dtype=int)
            steps_s = [steps[i] for i in idx]
            cds_s = [cds[i] for i in idx]

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(steps_s, cds_s, alpha=0.4, color='blue', linewidth=0.5)
            # Smoothed
            window = min(100, len(cds_s) // 5 + 1)
            if window > 1:
                smoothed = np.convolve(cds_s, np.ones(window)/window, mode='valid')
                ax.plot(steps_s[window-1:], smoothed, color='darkblue', linewidth=2, label='Smoothed')
            ax.set_xlabel('Training Step', fontsize=12)
            ax.set_ylabel('Chamfer Distance', fontsize=12)
            ax.set_title('Training Loss (2048-pt Retrain)', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'training_loss_curve.png'), dpi=150)
            plt.close()
            print(f"Saved training_loss_curve.png ({len(steps_s)} points)")

    # --- Results summary bar chart ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Data from our evaluations
    old_results = {'CD': 0.015218, 'F@0.01': 0.0196, 'F@0.02': 0.1211, 'F@0.05': 0.5865}
    new_results = {'CD': 0.009477, 'F@0.01': 0.0422, 'F@0.02': 0.1958, 'F@0.05': 0.6615}

    # CD comparison
    methods = ['Old (1024pts\nEpoch 100)', 'New (2048pts\nEpoch 40)']
    cd_vals = [old_results['CD'], new_results['CD']]
    bars = axes[0].bar(methods, cd_vals, color=['steelblue', 'coral'], width=0.5)
    axes[0].set_ylabel('Chamfer Distance (lower is better)', fontsize=11)
    axes[0].set_title('Chamfer Distance: Old vs New', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, cd_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.0003,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    # F-Score comparison
    x = np.arange(3)
    width = 0.35
    f_old = [old_results['F@0.01'], old_results['F@0.02'], old_results['F@0.05']]
    f_new = [new_results['F@0.01'], new_results['F@0.02'], new_results['F@0.05']]
    axes[1].bar(x - width/2, f_old, width, label='Old (1024pts)', color='steelblue')
    axes[1].bar(x + width/2, f_new, width, label='New (2048pts)', color='coral')
    axes[1].set_ylabel('F-Score (higher is better)', fontsize=11)
    axes[1].set_title('F-Score Comparison', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['F@0.01', 'F@0.02', 'F@0.05'])
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'results_comparison.png'), dpi=150)
    plt.close()
    print("Saved results_comparison.png")

    print(f"\nAll plots saved to {save_dir}/")


if __name__ == '__main__':
    main()
