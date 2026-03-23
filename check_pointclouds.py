"""
Check ground truth point cloud quality.
Visualizes raw .npy files to see if they look like proper 3D shapes.

Usage: python check_pointclouds.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import os

PC_DIR = "D:/DL/data/cap3d/point_clouds"
OUTPUT_DIR = "./visualizations/gt_check"
os.makedirs(OUTPUT_DIR, exist_ok=True)

pc_dir = Path(PC_DIR)
npy_files = sorted(pc_dir.glob("*.npy"))[:10]

print(f"Checking {len(npy_files)} point clouds from {PC_DIR}")

for i, npy_path in enumerate(npy_files):
    pts = np.load(str(npy_path))
    print(f"\n{npy_path.stem}:")
    print(f"  Shape: {pts.shape}")
    print(f"  Range X: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]")
    print(f"  Range Y: [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]")
    print(f"  Range Z: [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]")
    
    # Check for clustering (are points spread out or in tight clusters?)
    from scipy.spatial.distance import pdist
    if len(pts) > 100:
        sample = pts[np.random.choice(len(pts), 100, replace=False)]
        dists = pdist(sample)
        print(f"  Avg pairwise dist: {dists.mean():.3f}")
        print(f"  Min pairwise dist: {dists.min():.6f}")
        print(f"  Std of points: {pts.std(axis=0).mean():.3f}")
    
    # Plot
    fig = plt.figure(figsize=(12, 5))
    for j, (elev, azim) in enumerate([(30, 45), (0, 0), (90, 0)]):
        ax = fig.add_subplot(1, 3, j+1, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.5, alpha=0.6)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"View {j+1}")
    
    plt.suptitle(f"{npy_path.stem}", fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"gt_{i:02d}_{npy_path.stem[:12]}.png")
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"  Saved: {save_path}")

print(f"\nAll saved to {OUTPUT_DIR}/")
