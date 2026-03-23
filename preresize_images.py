"""
Pre-resize Cap3D renders from 512x512 RGBA PNG to 224x224 RGB JPG.

Runs once, makes training ~10x faster.

Input:  L:/DL/data/shapenet/renders/<synset>/<model>/image_XXXX.png
Output: G:/DL/data/shapenet/renders/<synset>/<model>/image_XXXX.jpg

Also copies point clouds to G: drive.

Usage:
    python preresize_images.py                 # full run
    python preresize_images.py --max 100       # test with 100
"""

import os
import shutil
import argparse
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

SRC_RENDERS = "D:/DL/data/shapenet/renders"
DST_RENDERS = "D:/DL/data/shapenet/renders_224"
SRC_PC = "D:/DL/data/cap3d/point_clouds"
DST_PC = "D:/DL/data/cap3d/point_clouds"
IMG_SIZE = 224


def resize_one_model(src_dir, dst_dir):
    """Resize all images in one model folder."""
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for png in sorted(Path(src_dir).glob("image_*.png")):
        jpg_name = png.stem + ".jpg"
        dst_path = dst_dir / jpg_name

        if dst_path.exists():
            count += 1
            continue

        try:
            img = Image.open(str(png)).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            img.save(str(dst_path), "JPEG", quality=95)
            count += 1
        except Exception as e:
            print(f"  Error: {png} — {e}")

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None, help="Process only N models (test)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    # Step 1: Resize renders
    print(f"Source renders: {SRC_RENDERS}")
    print(f"Dest renders:   {DST_RENDERS}")
    print(f"Resize to:      {IMG_SIZE}x{IMG_SIZE} RGB JPG")
    print()

    src_path = Path(SRC_RENDERS)
    if not src_path.exists():
        print(f"ERROR: {SRC_RENDERS} not found!")
        return

    # Collect all model dirs
    tasks = []
    for synset in sorted(src_path.iterdir()):
        if not synset.is_dir():
            continue
        for model in sorted(synset.iterdir()):
            if not model.is_dir():
                continue
            dst = Path(DST_RENDERS) / synset.name / model.name
            tasks.append((str(model), str(dst)))

    if args.max:
        tasks = tasks[:args.max]

    print(f"Models to process: {len(tasks)}")

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(resize_one_model, s, d): s for s, d in tasks}
        for future in as_completed(futures):
            done += 1
            if done % 2000 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}] resized")

    print(f"\nRenders done! {done} models resized.")

    # Step 2: Copy point clouds
    print(f"\nCopying point clouds...")
    print(f"  From: {SRC_PC}")
    print(f"  To:   {DST_PC}")

    os.makedirs(DST_PC, exist_ok=True)
    npy_files = list(Path(SRC_PC).glob("*.npy"))
    copied = 0
    skipped = 0
    for npy in npy_files:
        dst = Path(DST_PC) / npy.name
        if not dst.exists():
            shutil.copy2(str(npy), str(dst))
            copied += 1
        else:
            skipped += 1

    print(f"  Copied: {copied}, Skipped: {skipped}")
    print(f"  Total .npy on G: {len(list(Path(DST_PC).glob('*.npy')))}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  DONE!")
    print(f"{'='*60}")
    print(f"  Now update config.yaml:")
    print(f"    shapenet_root: G:/DL/data/shapenet")
    print(f"    cap3d_root: G:/DL/data/cap3d")
    print(f"  Then run: python train.py --config config.yaml")


if __name__ == "__main__":
    main()
