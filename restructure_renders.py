"""
Restructure Cap3D images into the folder layout that dataset.py expects.

Current:  images/02691156_1a04e3eab45ca15dd86060f189eb133/00000.png
Expected: renders/02691156/1a04e3eab45ca15dd86060f189eb133/image_0000.png

Also verifies the match between renders and point clouds.

Usage:
    python restructure_renders.py                  # full run
    python restructure_renders.py --max 100        # test with 100
    python restructure_renders.py --verify-only    # just check counts
"""

import os
import argparse
from pathlib import Path

IMAGES_DIR = "L:/DL/datasets/shapenet/images"
RENDERS_DIR = "L:/DL/data/shapenet/renders"
PC_DIR = "L:/DL/data/cap3d/point_clouds"


def restructure(images_dir, renders_dir, max_objects=None):
    images_path = Path(images_dir)
    renders_path = Path(renders_dir)

    obj_dirs = sorted([d for d in images_path.iterdir() if d.is_dir()])
    if max_objects:
        obj_dirs = obj_dirs[:max_objects]

    print(f"Objects to restructure: {len(obj_dirs)}")
    print(f"Source: {images_dir}")
    print(f"Target: {renders_dir}")
    print()

    created = 0
    skipped = 0

    for i, obj_dir in enumerate(obj_dirs):
        name = obj_dir.name
        parts = name.split("_", 1)
        if len(parts) != 2:
            continue

        synset_id, model_id = parts
        target_dir = renders_path / synset_id / model_id
        
        # Skip if already done
        if target_dir.exists() and any(target_dir.glob("image_*.png")):
            skipped += 1
            continue

        target_dir.mkdir(parents=True, exist_ok=True)

        # Find all RGB pngs (exclude depth and MatAlpha)
        rgb_pngs = sorted([
            f for f in obj_dir.glob("*.png")
            if "_depth" not in f.name and "_MatAlpha" not in f.name
        ])

        for j, src_png in enumerate(rgb_pngs):
            dst_name = f"image_{j:04d}.png"
            dst_path = target_dir / dst_name

            if not dst_path.exists():
                # Try symlink first (fast), fall back to copy
                try:
                    dst_path.symlink_to(src_png.resolve())
                except OSError:
                    import shutil
                    shutil.copy2(str(src_png), str(dst_path))

        created += 1

        if (i + 1) % 5000 == 0:
            print(f"  [{i+1}/{len(obj_dirs)}] created={created}, skipped={skipped}")

    print(f"\nDone restructuring!")
    print(f"  Created: {created}")
    print(f"  Skipped: {skipped}")


def verify(renders_dir, pc_dir):
    renders_path = Path(renders_dir)
    pc_path = Path(pc_dir)

    print(f"\n{'='*60}")
    print(f"  VERIFICATION")
    print(f"{'='*60}")

    # Count renders
    render_models = set()
    synset_counts = {}
    if renders_path.exists():
        for synset in renders_path.iterdir():
            if synset.is_dir():
                models = [m.name for m in synset.iterdir() if m.is_dir()]
                render_models.update(models)
                synset_counts[synset.name] = len(models)

    print(f"  Render models: {len(render_models)}")

    # Count point clouds
    pc_models = set()
    if pc_path.exists():
        pc_models = set(p.stem for p in pc_path.glob("*.npy"))
    print(f"  Point cloud .npy: {len(pc_models)}")

    # Match
    matched = render_models & pc_models
    print(f"  Matched (usable): {len(matched)}")

    if matched:
        print(f"\n  Per-category breakdown:")
        synset_names = {
            "02691156": "airplane", "02828884": "bench", "02933112": "cabinet",
            "02958343": "car", "03001627": "chair", "03211117": "display",
            "03636649": "lamp", "03691459": "loudspeaker", "04090263": "rifle",
            "04256520": "sofa", "04379243": "table", "04401088": "telephone",
            "04530566": "watercraft",
        }
        for synset_id, count in sorted(synset_counts.items(), key=lambda x: -x[1]):
            # Count how many of this synset's models have matching PCs
            synset_path = renders_path / synset_id
            synset_models = set(m.name for m in synset_path.iterdir() if m.is_dir())
            synset_matched = synset_models & pc_models
            name = synset_names.get(synset_id, synset_id)
            print(f"    {name:15s} ({synset_id}): {len(synset_matched):5d} matched / {count} rendered")

        print(f"\n  Ready for training with {len(matched)} objects!")
    else:
        print(f"\n  WARNING: No matches found!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default=IMAGES_DIR)
    parser.add_argument("--renders-dir", default=RENDERS_DIR)
    parser.add_argument("--pc-dir", default=PC_DIR)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if not args.verify_only:
        restructure(args.images_dir, args.renders_dir, args.max)

    verify(args.renders_dir, args.pc_dir)


if __name__ == "__main__":
    main()
