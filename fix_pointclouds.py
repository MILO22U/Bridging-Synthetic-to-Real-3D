"""
Fix point clouds: generate from SINGLE best depth view instead of merging.

The multi-view merge was broken because camera extrinsics weren't aligning.
Single-view gives a clean half-shape which is actually what the model should
learn to complete.

Usage:
    python fix_pointclouds.py --max 100      # test
    python fix_pointclouds.py                # all
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
BG_THRESHOLD = 64000


def backproject_single_view(obj_dir, num_points=1024):
    """
    Back-project the BEST single depth view into a point cloud.
    No multi-view merging — avoids the alignment bug.
    Points stay in camera-local space (no world transform needed).
    """
    transforms_path = os.path.join(obj_dir, "transforms_train.json")
    if not os.path.exists(transforms_path):
        return None

    with open(transforms_path) as f:
        tf = json.load(f)

    fov_x = tf.get("camera_angle_x", 0.6911)
    frames = tf.get("frames", [])
    if not frames:
        return None

    # Try multiple views, pick the one with most foreground pixels
    best_pts = None
    best_count = 0

    for vi in range(min(len(frames), 20)):
        depth_path = os.path.join(obj_dir, f"{vi:05d}_depth.png")
        cam_path = os.path.join(obj_dir, f"{vi:05d}.json")

        if not os.path.exists(depth_path):
            continue

        try:
            depth_img = np.array(Image.open(depth_path)).astype(np.float32)
        except:
            continue

        # Read max_depth from per-view camera json
        max_depth = 5.0
        if os.path.exists(cam_path):
            try:
                with open(cam_path) as f:
                    cam = json.load(f)
                max_depth = cam.get("max_depth", 5.0)
                fov_x = cam.get("x_fov", fov_x)
            except:
                pass

        # Foreground mask
        fg_mask = depth_img < BG_THRESHOLD
        fg_count = fg_mask.sum()

        if fg_count < 200 or fg_count <= best_count:
            continue

        # Convert to metric depth
        depth = depth_img / 65535.0 * max_depth

        H, W = depth.shape
        fx = W / (2.0 * np.tan(fov_x / 2.0))
        fy = fx
        cx, cy = W / 2.0, H / 2.0

        # Back-project to camera space (NOT world space)
        v, u = np.where(fg_mask)
        z = depth[v, u]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        pts = np.stack([x, y, z], axis=-1).astype(np.float32)
        best_pts = pts
        best_count = fg_count

    if best_pts is None or len(best_pts) < 100:
        return None

    return best_pts


def process_one(obj_dir, output_dir, num_points=1024):
    """Process single object."""
    obj_name = os.path.basename(obj_dir)
    parts = obj_name.split("_", 1)
    if len(parts) != 2:
        return (obj_name, 0)

    synset_id, model_id = parts
    out_path = os.path.join(output_dir, f"{model_id}.npy")

    pts = backproject_single_view(obj_dir, num_points)
    if pts is None:
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
    parser.add_argument("--images-dir", default=IMAGES_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--num-points", type=int, default=NUM_POINTS)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    obj_dirs = sorted([d for d in Path(args.images_dir).iterdir() if d.is_dir()])
    if args.max:
        obj_dirs = obj_dirs[:args.max]

    print(f"Objects: {len(obj_dirs)}")
    print(f"Output: {args.output_dir}")
    print(f"Method: single best view (camera space)")
    print()

    success = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one, str(d), args.output_dir, args.num_points): d.name
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


if __name__ == "__main__":
    main()
