"""
Check what's in L:\DL\datasets\shapenet\images\ 
and retry Cap3D_pcs.npz download.

Run: python check_images_and_retry.py
"""

import os
from pathlib import Path

# ── Check images/ folder ──────────────────────────────
print("=" * 60)
print("[1] Checking L:\\DL\\datasets\\shapenet\\images\\")
print("=" * 60)

img_dir = Path(r"L:\DL\datasets\shapenet\images")
if img_dir.exists():
    items = list(img_dir.iterdir())
    print(f"  Total items: {len(items)}")
    
    # Show first 5
    dirs = 0
    files = 0
    for item in items[:5]:
        if item.is_dir():
            contents = list(item.iterdir())
            print(f"  [DIR]  {item.name}/  ({len(contents)} items inside)")
            # Show what's inside
            for c in contents[:3]:
                print(f"         {c.name}")
            dirs += 1
        else:
            size_kb = item.stat().st_size / 1024
            print(f"  [FILE] {item.name}  ({size_kb:.0f} KB)")
            files += 1
    
    # Count totals
    all_dirs = sum(1 for i in items if i.is_dir())
    all_files = sum(1 for i in items if i.is_file())
    print(f"\n  Directories: {all_dirs}")
    print(f"  Files: {all_files}")
    
    # Check if these are synset_modelid folders or something else
    sample_names = [i.name for i in items[:10]]
    print(f"  Sample names: {sample_names}")
else:
    print("  NOT FOUND")

# ── Check renders/ folder ─────────────────────────────
print("\n" + "=" * 60)
print("[2] Checking L:\\DL\\datasets\\shapenet\\renders\\")
print("=" * 60)

renders_dir = Path(r"L:\DL\datasets\shapenet\renders")
if renders_dir.exists():
    for item in sorted(renders_dir.iterdir()):
        if item.is_dir():
            count = sum(1 for _ in item.iterdir())
            print(f"  [DIR]  {item.name}/  ({count} items)")
        else:
            size_mb = item.stat().st_size / 1e6
            print(f"  [FILE] {item.name}  ({size_mb:.1f} MB)")

# ── Check Cap3D_pcs.npz ──────────────────────────────
print("\n" + "=" * 60)
print("[3] Checking Cap3D_pcs.npz")
print("=" * 60)

npz_path = Path(r"L:\DL\data\cap3d\Cap3D_pcs.npz")
if npz_path.exists():
    size = npz_path.stat().st_size
    print(f"  Size: {size} bytes ({size/1e6:.1f} MB)")
    if size < 1000:
        print("  ⚠  File is essentially empty — download failed")
        print("  Deleting it so we can retry...")
        os.remove(str(npz_path))
        print("  Deleted.")
else:
    print("  Not found")

print("\n" + "=" * 60)
print("Paste this output to me.")
