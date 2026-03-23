"""
Generate proper 3D point clouds by merging multiple depth views
using camera extrinsics from per-view JSON files.

Each JSON has: origin, x/y/z axes, x_fov, y_fov, max_depth, bbox
We back-project each depth map and transform to world coordinates.

Usage:
    python fix_pointclouds_multiview.py --max 100    # test
    python fix_pointclouds_multiview.py               # all
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

IMAGES_DIR = "D:/DL/datasets/shapenet/images"
OUTPUT_DIR = "D:/DL/data/cap3d/point_clouds"
NUM_POINTS = 1024
NUM_VIEWS = 8  # Use 8 best views (not all 20 — too many overlapping points)
BG_THRESHOLD = 64000


def backproject_view(depth_path, cam_json_path):
    """Back-project a single depth map to 3D world coordinates."""
    if not os.path.exists(depth_path) or not os.path.exists(cam_json_path):
        return None

    try:
        with open(cam_json_path) as f:
            cam = json.load(f)
    except:
        return None

    try:
        depth_img = np.array(Image.open(depth_path)).astype(np.float32)
    except:
        return None

    max_depth = cam.get("max_depth", 5.0)
    x_fov = cam.get("x_fov", 0.6911)
    y_fov = cam.get("y_fov", x_fov)

    # Camera axes and origin
    origin = np.array(cam["origin"])
    cam_x = np.array(cam["x"])  # right
    cam_y = np.array(cam["y"])  # down
    cam_z = np.array(cam["z"])  # forward (into scene)

    # Foreground mask
    fg_mask = depth_img < BG_THRESHOLD
    if fg_mask.sum() < 50:
        return None

    # Convert to metric depth
    depth = depth_img / 65535.0 * max_depth

    H, W = depth.shape
    fx = W / (2.0 * np.tan(x_fov / 2.0))
    fy = H / (2.0 * np.tan(y_fov / 2.0))
    cx, cy = W / 2.0, H / 2.0

    # Get foreground pixel coordinates
    v, u = np.where(fg_mask)
    z = depth[v, u]

    # Back-project to camera space
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    z_cam = z

    # Transform to world space using camera axes
    # Point in world = origin + x_cam * cam_x + y_cam * cam_y + z_cam * cam_z
    pts_world = (
        origin[np.newaxis, :] +
        x_cam[:, np.newaxis] * cam_x[np.newaxis, :] +
        y_cam[:, np.newaxis] * cam_y[np.newaxis, :] +
        z_cam[:, np.newaxis] * cam_z[np.newaxis, :]
    )

    return pts_world.astype(np.float32)


def process_one_object(obj_dir, output_dir, num_points=1024, num_views=8):
    """Process one object: merge multiple views into single point cloud."""
    obj_name = os.path.basename(obj_dir)
    parts = obj_name.split("_", 1)
    if len(parts) != 2:
        return (obj_name, 0)

    synset_id, model_id = parts
    out_path = os.path.join(output_dir, f"{model_id}.npy")

    # Collect points from multiple views
    all_points = []
    view_scores = []

    # First pass: score views by foreground pixel count
    for vi in range(20):
        depth_path = os.path.join(obj_dir, f"{vi:05d}_depth.png")
        if os.path.exists(depth_path):
            try:
                d = np.array(Image.open(depth_path)).astype(np.float32)
                fg = (d < BG_THRESHOLD).sum()
                view_scores.append((vi, fg))
            except:
                pass

    # Sort by foreground count, take top N views
    view_scores.sort(key=lambda x: -x[1])
    best_views = [vs[0] for vs in view_scores[:num_views]]

    for vi in best_views:
        depth_path = os.path.join(obj_dir, f"{vi:05d}_depth.png")
        cam_path = os.path.join(obj_dir, f"{vi:05d}.json")
        pts = backproject_view(depth_path, cam_path)
        if pts is not None and len(pts) > 0:
            all_points.append(pts)

    if not all_points:
        return (model_id, 0)

    # Merge all views
    merged = np.concatenate(all_points, axis=0)

    if len(merged) < 100:
        return (model_id, 0)

    # Normalize: center on bbox center, scale to [-1, 1]
    # Read bbox from first view's JSON
    cam_path = os.path.join(obj_dir, f"{best_views[0]:05d}.json")
    try:
        with open(cam_path) as f:
            cam = json.load(f)
        bbox = np.array(cam.get("bbox", [[-1,-1,-1],[1,1,1]]))
        center = (bbox[0] + bbox[1]) / 2.0
        scale = np.max(bbox[1] - bbox[0]) / 2.0
    except:
        center = merged.mean(axis=0)
        scale = np.max(np.linalg.norm(merged - center, axis=1))

    merged = (merged - center) / max(scale, 1e-6)

    # Remove outliers (points far from center)
    dists = np.linalg.norm(merged, axis=1)
    mask = dists < 3.0  # Keep points within 3 units
    merged = merged[mask]

    if len(merged) < 100:
        return (model_id, 0)

    # Voxel downsample to remove duplicates from overlapping views
    voxel_size = 0.02
    voxel_indices = np.floor(merged / voxel_size).astype(int)
    _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
    merged = merged[unique_idx]

    # Subsample to target number of points
    if len(merged) > num_points:
        idx = np.random.choice(len(merged), num_points, replace=False)
        merged = merged[idx]
    elif len(merged) < num_points:
        idx = np.random.choice(len(merged), num_points, replace=True)
        merged = merged[idx]

    # Final normalize to [-1, 1]
    centroid = merged.mean(axis=0)
    merged = merged - centroid
    max_dist = np.max(np.linalg.norm(merged, axis=1))
    if max_dist > 0:
        merged = merged / max_dist

    np.save(out_path, merged.astype(np.float32))
    return (model_id, len(merged))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default=IMAGES_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--num-points", type=int, default=NUM_POINTS)
    parser.add_argument("--num-views", type=int, default=NUM_VIEWS)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    obj_dirs = sorted([d for d in Path(args.images_dir).iterdir() if d.is_dir()])
    if args.max:
        obj_dirs = obj_dirs[:args.max]

    print(f"Objects: {len(obj_dirs)}")
    print(f"Output: {args.output_dir}")
    print(f"Method: multi-view merge ({args.num_views} best views)")
    print(f"Points: {args.num_points}")
    print()

    success = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_one_object, str(d), args.output_dir,
                args.num_points, args.num_views
            ): d.name
            for d in obj_dirs
        }
        for i, future in enumerate(as_completed(futures)):
            model_id, n = future.result()
            if n > 0:
                success += 1
            else:
                failed += 1
            if (i + 1) % 5000 == 0 or (i + 1) == len(obj_dirs):
                print(f"  [{i+1}/{len(obj_dirs)}] success={success}, failed={failed}")

    print(f"\nDone! success={success}, failed={failed}")
    print(f"\nNow retrain: python train.py --config config.yaml")


if __name__ == "__main__":
    main()
