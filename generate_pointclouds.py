"""
Generate 3D point clouds from Cap3D depth maps + camera matrices.

Back-projects depth maps from multiple views into world-space 3D points,
merges them, and subsamples to 2048 points per object.

Input:  L:\DL\datasets\shapenet\images\<synset>_<model_id>\  (20 views each)
Output: L:\DL\data\cap3d\point_clouds\<model_id>.npy          (2048 x 3 float32)

Usage:
    python generate_pointclouds.py                          # all 45,000 objects
    python generate_pointclouds.py --max 100                # first 100 only (test)
    python generate_pointclouds.py --views 5                # use 5 views per object (faster)
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed


# ─── Config ───────────────────────────────────────────
IMAGES_DIR = r"L:/DL/datasets/shapenet/images"
OUTPUT_DIR = r"L:/DL/data/cap3d/point_clouds"
NUM_POINTS = 2048     # points per object
NUM_VIEWS = 8         # views to merge (more = better, slower; 8 is good balance)
BG_THRESHOLD = 64000  # depth uint16 values above this are background


def backproject_depth(depth_path, cam_json_path, transforms_json_path, frame_idx):
    """
    Back-project a single depth map to 3D world-space points.

    Returns: (N, 3) numpy array of 3D points, or None on failure
    """
    try:
        depth_img = np.array(Image.open(depth_path)).astype(np.float32)
    except Exception:
        return None

    try:
        with open(cam_json_path) as f:
            cam = json.load(f)
        with open(transforms_json_path) as f:
            tf = json.load(f)
    except Exception:
        return None

    max_depth = cam.get("max_depth", 5.0)
    fov_x = cam.get("x_fov", tf.get("camera_angle_x", 0.6911))

    # Convert uint16 depth to metric
    depth = depth_img / 65535.0 * max_depth

    # Foreground mask
    fg_mask = depth_img < BG_THRESHOLD
    if fg_mask.sum() < 50:
        return None

    H, W = depth.shape
    fx = W / (2.0 * np.tan(fov_x / 2.0))
    fy = fx
    cx, cy = W / 2.0, H / 2.0

    # Back-project
    v, u = np.where(fg_mask)
    z = depth[v, u]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=-1)

    # Camera-to-world transform
    try:
        mat = np.array(tf["frames"][frame_idx]["transform_matrix"])
    except (IndexError, KeyError):
        return None

    R = mat[:3, :3]
    t = mat[:3, 3]
    pts_world = (R @ pts_cam.T).T + t

    return pts_world.astype(np.float32)


def process_one_object(obj_dir, output_dir, num_points=2048, num_views=8):
    """
    Process a single object: merge multiple depth views into one point cloud.

    Args:
        obj_dir: Path to e.g. images/02691156_1a04e3eab45ca15dd86060f189eb133/
        output_dir: Where to save .npy files
        num_points: Target point count
        num_views: How many views to merge

    Returns: (model_id, num_points_generated) or (model_id, 0) on failure
    """
    obj_name = os.path.basename(obj_dir)
    parts = obj_name.split("_", 1)
    if len(parts) != 2:
        return (obj_name, 0)

    synset_id, model_id = parts
    out_path = os.path.join(output_dir, f"{model_id}.npy")

    # Skip if already done
    if os.path.exists(out_path):
        return (model_id, -1)  # already exists

    transforms_path = os.path.join(obj_dir, "transforms_train.json")
    if not os.path.exists(transforms_path):
        return (model_id, 0)

    # Collect points from multiple views
    all_points = []
    # Use evenly spaced views for better coverage
    view_indices = np.linspace(0, 19, num_views, dtype=int)

    for vi in view_indices:
        depth_path = os.path.join(obj_dir, f"{vi:05d}_depth.png")
        cam_path = os.path.join(obj_dir, f"{vi:05d}.json")

        if not os.path.exists(depth_path) or not os.path.exists(cam_path):
            continue

        pts = backproject_depth(depth_path, cam_path, transforms_path, vi)
        if pts is not None and len(pts) > 0:
            all_points.append(pts)

    if not all_points:
        return (model_id, 0)

    merged = np.concatenate(all_points, axis=0)

    # Remove outliers (points far from center)
    centroid = merged.mean(axis=0)
    dists = np.linalg.norm(merged - centroid, axis=1)
    threshold = np.percentile(dists, 98)
    merged = merged[dists < threshold]

    if len(merged) < 100:
        return (model_id, 0)

    # Subsample to target count
    if len(merged) > num_points:
        idx = np.random.choice(len(merged), num_points, replace=False)
        final = merged[idx]
    else:
        # Upsample by repeating with small noise
        idx = np.random.choice(len(merged), num_points, replace=True)
        final = merged[idx]
        final += np.random.randn(*final.shape).astype(np.float32) * 0.001

    # Normalize to [-1, 1]
    centroid = final.mean(axis=0)
    final = final - centroid
    max_dist = np.max(np.linalg.norm(final, axis=1))
    if max_dist > 0:
        final = final / max_dist

    np.save(out_path, final.astype(np.float32))
    return (model_id, len(final))


def main():
    parser = argparse.ArgumentParser(description="Generate point clouds from depth maps")
    parser.add_argument("--images-dir", default=IMAGES_DIR,
                        help="Directory with extracted Cap3D folders")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Output directory for .npy files")
    parser.add_argument("--num-points", type=int, default=NUM_POINTS)
    parser.add_argument("--views", type=int, default=NUM_VIEWS,
                        help="Views to merge per object (more=better, slower)")
    parser.add_argument("--max", type=int, default=None,
                        help="Process only first N objects (for testing)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all object folders
    images_path = Path(args.images_dir)
    obj_dirs = sorted([d for d in images_path.iterdir() if d.is_dir()])
    if args.max:
        obj_dirs = obj_dirs[:args.max]

    print(f"Objects to process: {len(obj_dirs)}")
    print(f"Views per object: {args.views}")
    print(f"Points per object: {args.num_points}")
    print(f"Output: {args.output_dir}")
    print(f"Workers: {args.workers}")
    print()

    success = 0
    skipped = 0
    failed = 0

    # Process with multiprocessing for speed
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_one_object,
                str(d), args.output_dir, args.num_points, args.views
            ): d.name
            for d in obj_dirs
        }

        for i, future in enumerate(as_completed(futures)):
            model_id, n_pts = future.result()
            if n_pts == -1:
                skipped += 1
            elif n_pts > 0:
                success += 1
            else:
                failed += 1

            if (i + 1) % 500 == 0 or (i + 1) == len(obj_dirs):
                total = success + skipped + failed
                print(f"  [{total}/{len(obj_dirs)}] "
                      f"success={success}, skipped={skipped}, failed={failed}")

    print(f"\nDone!")
    print(f"  Success: {success}")
    print(f"  Skipped (already existed): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Output: {args.output_dir}")

    # Quick verify
    npy_count = len(list(Path(args.output_dir).glob("*.npy")))
    print(f"  Total .npy files: {npy_count}")


if __name__ == "__main__":
    main()
