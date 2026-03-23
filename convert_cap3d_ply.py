"""
Convert Cap3D ShapeNet PLY point clouds to NPY format.

PLY files have 16384 points with xyz+rgb. We subsample to num_points
and save as (N, 3) float32 NPY files.

Usage:
    python convert_cap3d_ply.py --max 100    # test
    python convert_cap3d_ply.py              # all
"""

import os
import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


PLY_DIR = "D:/DL/data/cap3d_ply/shapenet_pcs"
OUTPUT_DIR = "D:/DL/data/cap3d/point_clouds"
NUM_POINTS = 2048


def read_ply_points(ply_path):
    """Read XYZ points from a PLY file."""
    with open(ply_path, 'rb') as f:
        # Read header
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            if line == 'end_header':
                break

        # Read binary data — 6 floats per vertex (x,y,z,r,g,b)
        data = np.frombuffer(f.read(), dtype=np.float32)
        data = data.reshape(-1, 6)
        xyz = data[:, :3]  # Just xyz

    return xyz


def process_one(ply_path, output_dir, num_points):
    """Convert one PLY to NPY."""
    name = Path(ply_path).stem  # e.g. 02691156_10155655850468db78d106ce0a280f87
    parts = name.split("_", 1)
    if len(parts) != 2:
        return (name, 0)

    model_id = parts[1]
    out_path = os.path.join(output_dir, f"{model_id}.npy")

    try:
        pts = read_ply_points(ply_path)
    except Exception as e:
        return (model_id, 0)

    if len(pts) < 100:
        return (model_id, 0)

    # Subsample
    if len(pts) > num_points:
        idx = np.random.choice(len(pts), num_points, replace=False)
        pts = pts[idx]
    elif len(pts) < num_points:
        idx = np.random.choice(len(pts), num_points, replace=True)
        pts = pts[idx]

    # Normalize to [-1, 1]
    centroid = pts.mean(axis=0)
    pts = pts - centroid
    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 0:
        pts = pts / max_dist

    np.save(out_path, pts.astype(np.float32))
    return (model_id, len(pts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply-dir", default=PLY_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--num-points", type=int, default=NUM_POINTS)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ply_files = sorted(Path(args.ply_dir).glob("*.ply"))
    if args.max:
        ply_files = ply_files[:args.max]

    print(f"PLY files: {len(ply_files)}")
    print(f"Output: {args.output_dir}")
    print(f"Points per cloud: {args.num_points}")
    print()

    success = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one, str(f), args.output_dir, args.num_points): f.name
            for f in ply_files
        }
        for i, future in enumerate(as_completed(futures)):
            model_id, n = future.result()
            if n > 0:
                success += 1
            else:
                failed += 1
            if (i + 1) % 5000 == 0 or (i + 1) == len(ply_files):
                print(f"  [{i+1}/{len(ply_files)}] success={success}, failed={failed}")

    print(f"\nDone! success={success}, failed={failed}")
    print(f"Now retrain: python train.py --config config.yaml")


if __name__ == "__main__":
    main()
