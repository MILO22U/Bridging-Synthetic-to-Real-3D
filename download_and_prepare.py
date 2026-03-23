"""
Download & Prepare Data for Synthetic-to-Real 3D Reconstruction.

This script ACTUALLY downloads and processes the data (not just prints instructions).

What it does:
  Step A: Download ShapeNet R2N2 renderings (24 views per object, 13 categories)
  Step B: Download Cap3D point clouds from Hugging Face and convert to .npy
  Step C: Create the folder structure that dataset.py expects

Required folder structure (what dataset.py reads):
  data/
    shapenet/
      renders/
        02691156/          (airplane synset)
          <model_id>/
            image_0000.png ... image_0023.png
        03001627/          (chair synset)
          ...
    cap3d/
      point_clouds/
        <model_id>.npy     (16384 x 3 or 16384 x 6)

Usage:
    python download_and_prepare.py --step a        # ShapeNet renders only
    python download_and_prepare.py --step b        # Cap3D point clouds only
    python download_and_prepare.py --step c        # Rename/link R2N2 to match dataset.py
    python download_and_prepare.py --step all      # Everything
"""

import os
import sys
import argparse
import subprocess
import glob
import shutil
import numpy as np
from pathlib import Path


DATA_ROOT = "./data"

# ShapeNet synset IDs
SYNSETS = {
    "02691156": "airplane",
    "02828884": "bench",
    "02933112": "cabinet",
    "02958343": "car",
    "03001627": "chair",
    "03211117": "display",
    "03636649": "lamp",
    "03691459": "loudspeaker",
    "04090263": "rifle",
    "04256520": "sofa",
    "04379243": "table",
    "04401088": "telephone",
    "04530566": "watercraft",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step A: ShapeNet R2N2 Renderings
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def step_a_shapenet_renders(data_root):
    """
    Download ShapeNet renderings from the 3D-R2N2 project.
    ~7.5 GB download, extracts to ShapeNetRendering/ folder.
    """
    shapenet_dir = Path(data_root) / "shapenet"
    shapenet_dir.mkdir(parents=True, exist_ok=True)

    tgz_path = shapenet_dir / "ShapeNetRendering.tgz"
    extracted_dir = shapenet_dir / "ShapeNetRendering"

    # Check if already extracted
    if extracted_dir.exists() and any(extracted_dir.iterdir()):
        count = sum(1 for d in extracted_dir.iterdir() if d.is_dir())
        print(f"[Step A] ShapeNet renderings already exist: {extracted_dir} ({count} categories)")
        return

    # Download
    url = "http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz"
    print(f"[Step A] Downloading ShapeNet renderings (~7.5 GB)...")
    print(f"  URL: {url}")
    print(f"  Saving to: {tgz_path}")
    print()

    if not tgz_path.exists():
        try:
            subprocess.run(
                ["wget", "-c", "--progress=bar:force", url, "-O", str(tgz_path)],
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("  wget failed. Trying curl...")
            subprocess.run(
                ["curl", "-L", "-C", "-", "-o", str(tgz_path), url],
                check=True,
            )
    else:
        print(f"  Archive already downloaded: {tgz_path}")

    # Extract
    print(f"  Extracting to {shapenet_dir}/ ...")
    subprocess.run(
        ["tar", "-xzf", str(tgz_path), "-C", str(shapenet_dir)],
        check=True,
    )
    print(f"  Done! Extracted to {extracted_dir}")

    # Count what we got
    for synset_id, name in SYNSETS.items():
        synset_path = extracted_dir / synset_id
        if synset_path.exists():
            n_models = sum(1 for d in synset_path.iterdir() if d.is_dir())
            print(f"    {synset_id} ({name}): {n_models} models")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step B: Cap3D Point Clouds
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def step_b_cap3d_pointclouds(data_root):
    """
    Download Cap3D point clouds from Hugging Face and convert to .npy files.

    Cap3D provides 16,384 colored points per ShapeNet object.
    We download the .pt files and convert each to <model_id>.npy.
    """
    cap3d_dir = Path(data_root) / "cap3d"
    pc_dir = cap3d_dir / "point_clouds"
    raw_dir = cap3d_dir / "raw"
    cap3d_dir.mkdir(parents=True, exist_ok=True)
    pc_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    existing_npy = list(pc_dir.glob("*.npy"))
    if len(existing_npy) > 100:
        print(f"[Step B] Cap3D point clouds already exist: {len(existing_npy)} .npy files in {pc_dir}")
        return

    print(f"[Step B] Downloading Cap3D point clouds from Hugging Face...")
    print()

    # Try huggingface_hub first
    try:
        from huggingface_hub import hf_hub_download
        print("  Using huggingface_hub to download Cap3D_pcs.npz ...")
        npz_path = hf_hub_download(
            repo_id="tiange/Cap3D",
            filename="misc/Cap3D_pcs.npz",
            repo_type="dataset",
            local_dir=str(cap3d_dir),
        )
        print(f"  Downloaded to: {npz_path}")
        _convert_npz_to_npy(npz_path, pc_dir)
        return

    except ImportError:
        print("  huggingface_hub not installed. Trying wget fallback...")
    except Exception as e:
        print(f"  huggingface_hub download failed: {e}")
        print("  Trying wget fallback...")

    # Fallback: wget the npz directly
    npz_url = "https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/Cap3D_pcs.npz"
    npz_path = cap3d_dir / "Cap3D_pcs.npz"

    if not npz_path.exists():
        print(f"  Downloading {npz_url}")
        print(f"  This file is large (~2-4 GB). Please be patient.")
        try:
            subprocess.run(
                ["wget", "-c", "--progress=bar:force", npz_url, "-O", str(npz_path)],
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            subprocess.run(
                ["curl", "-L", "-C", "-", "-o", str(npz_path), npz_url],
                check=True,
            )

    _convert_npz_to_npy(str(npz_path), pc_dir)


def _convert_npz_to_npy(npz_path, output_dir):
    """Convert a single Cap3D .npz file into per-model .npy files."""
    print(f"  Loading {npz_path} ...")
    data = np.load(npz_path, allow_pickle=True)

    keys = list(data.files) if hasattr(data, 'files') else list(data.keys())
    print(f"  Found {len(keys)} objects in npz")

    count = 0
    for key in keys:
        arr = data[key]
        # arr is typically (16384, 6) — xyz + rgb
        # Save only xyz (first 3 cols) or full array
        out_path = Path(output_dir) / f"{key}.npy"
        if not out_path.exists():
            np.save(str(out_path), arr)
            count += 1

    print(f"  Saved {count} new .npy files to {output_dir}")
    print(f"  Total .npy files: {len(list(Path(output_dir).glob('*.npy')))}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step C: Rename R2N2 structure to match dataset.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def step_c_restructure(data_root):
    """
    The 3D-R2N2 download extracts to:
      data/shapenet/ShapeNetRendering/<synset>/<model_id>/rendering/00.png ...

    But dataset.py expects:
      data/shapenet/renders/<synset>/<model_id>/image_0000.png ...

    This step creates symlinks (or copies) + renames files to match.
    """
    source_dir = Path(data_root) / "shapenet" / "ShapeNetRendering"
    target_dir = Path(data_root) / "shapenet" / "renders"

    if not source_dir.exists():
        print(f"[Step C] Source not found: {source_dir}")
        print("         Run Step A first.")
        return

    if target_dir.exists() and any(target_dir.iterdir()):
        count = sum(1 for _ in target_dir.rglob("image_0000.png"))
        if count > 0:
            print(f"[Step C] Renders already restructured: {target_dir} ({count} models)")
            return

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Step C] Restructuring R2N2 renders → dataset.py format")
    print(f"  Source: {source_dir}")
    print(f"  Target: {target_dir}")

    total_models = 0
    for synset_id in SYNSETS:
        synset_src = source_dir / synset_id
        synset_dst = target_dir / synset_id
        if not synset_src.exists():
            continue

        synset_dst.mkdir(exist_ok=True)

        for model_dir in synset_src.iterdir():
            if not model_dir.is_dir():
                continue

            model_id = model_dir.name
            model_dst = synset_dst / model_id
            model_dst.mkdir(exist_ok=True)

            # R2N2 puts images in <model>/rendering/00.png, 01.png, ...
            rendering_dir = model_dir / "rendering"
            if not rendering_dir.exists():
                # Some versions have images directly in model dir
                rendering_dir = model_dir

            pngs = sorted(rendering_dir.glob("*.png"))
            for i, png_path in enumerate(pngs):
                dst_name = f"image_{i:04d}.png"
                dst_path = model_dst / dst_name
                if not dst_path.exists():
                    # Use symlink for speed; falls back to copy
                    try:
                        dst_path.symlink_to(png_path.resolve())
                    except OSError:
                        shutil.copy2(str(png_path), str(dst_path))

            total_models += 1

        n = sum(1 for d in synset_dst.iterdir() if d.is_dir())
        print(f"    {synset_id} ({SYNSETS[synset_id]}): {n} models")

    print(f"  Total models restructured: {total_models}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Verify
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def verify(data_root):
    """Check that the data is ready for training."""
    render_dir = Path(data_root) / "shapenet" / "renders"
    pc_dir = Path(data_root) / "cap3d" / "point_clouds"

    print()
    print("=" * 60)
    print("  VERIFICATION")
    print("=" * 60)

    # Renders
    if render_dir.exists():
        render_models = set()
        for synset in render_dir.iterdir():
            if synset.is_dir():
                for model in synset.iterdir():
                    if model.is_dir():
                        render_models.add(model.name)
        print(f"  Rendered models : {len(render_models)}")
    else:
        render_models = set()
        print(f"  Rendered models : MISSING ({render_dir})")

    # Point clouds
    if pc_dir.exists():
        pc_models = set(p.stem for p in pc_dir.glob("*.npy"))
        print(f"  Point cloud .npy: {len(pc_models)}")
    else:
        pc_models = set()
        print(f"  Point cloud .npy: MISSING ({pc_dir})")

    # Overlap = usable samples
    overlap = render_models & pc_models
    print(f"  Matched (usable): {len(overlap)}")

    if len(overlap) == 0:
        print()
        print("  ⚠  No matched samples found!")
        print("  The model_id in renders must match the .npy filename in point_clouds/.")
        print("  Check if Cap3D uses Objaverse IDs vs ShapeNet IDs.")
        print()
        print("  If Cap3D IDs don't match ShapeNet model IDs, you have two options:")
        print("  1. Generate point clouds directly from ShapeNet meshes (see below)")
        print("  2. Use a ShapeNet↔Objaverse ID mapping file")
        print()
        print("  To generate from meshes:")
        print("    pip install trimesh")
        print("    python generate_pointclouds_from_meshes.py --shapenet_root data/shapenet")
    else:
        print()
        print(f"  ✓ Ready for training! {len(overlap)} matched objects.")
        # Show per-category breakdown
        for synset_id, name in SYNSETS.items():
            synset_path = render_dir / synset_id
            if synset_path.exists():
                synset_models = set(d.name for d in synset_path.iterdir() if d.is_dir())
                matched = synset_models & pc_models
                print(f"    {name:15s}: {len(matched)} matched")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="Download & prepare training data")
    parser.add_argument("--step", type=str, default="all",
                        choices=["a", "b", "c", "all", "verify"],
                        help="a=ShapeNet, b=Cap3D, c=restructure, all=everything")
    parser.add_argument("--data-root", type=str, default=DATA_ROOT,
                        help="Root data directory (default: ./data)")
    args = parser.parse_args()

    print(f"Data root: {os.path.abspath(args.data_root)}")
    print()

    if args.step in ("a", "all"):
        step_a_shapenet_renders(args.data_root)
        print()

    if args.step in ("b", "all"):
        step_b_cap3d_pointclouds(args.data_root)
        print()

    if args.step in ("c", "all"):
        step_c_restructure(args.data_root)
        print()

    # Always verify at the end
    verify(args.data_root)


if __name__ == "__main__":
    main()
