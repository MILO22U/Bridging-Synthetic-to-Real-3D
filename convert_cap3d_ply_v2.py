"""
Convert Cap3D ShapeNet PLY (ASCII) point clouds to NPY format.

Usage:
    python convert_cap3d_ply_v2.py --max 100
    python convert_cap3d_ply_v2.py
"""

import os
import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PLY_DIR = "D:/DL/data/cap3d_ply/shapenet_pcs/ShapeNet_pcs"
OUTPUT_DIR = "D:/DL/data/cap3d/point_clouds"
NUM_POINTS = 2048


def read_ply_ascii(ply_path):
    """Read XYZ from ASCII PLY file."""
    with open(ply_path, 'r') as f:
        num_vertices = 0
        while True:
            line = f.readline().strip()
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            if line == 'end_header':
                break

        points = []
        for _ in range(num_vertices):
            parts = f.readline().strip().split()
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            points.append([x, y, z])

    return np.array(points, dtype=np.float32)


def process_one(ply_path, output_dir, num_points):
    name = Path(ply_path).stem
    parts = name.split("_", 1)
    if len(parts) != 2:
        return (name, 0)

    model_id = parts[1]
    out_path = os.path.join(output_dir, f"{model_id}.npy")

    try:
        pts = read_ply_ascii(ply_path)
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
    print(f"Points: {args.num_points}")

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


if __name__ == "__main__":
    main()
