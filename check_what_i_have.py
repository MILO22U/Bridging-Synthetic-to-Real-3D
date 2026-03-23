"""
Run this on your Windows machine to check what you have.

Usage: python check_what_i_have.py
"""

import os
from pathlib import Path

print("=" * 60)
print("  CHECKING YOUR DISK FOR SHAPENET DATA")
print("=" * 60)

# ── Check 1: ShapeNet meshes ──────────────────────────
print("\n[1] Looking for ShapeNet meshes (.obj files)...")
search_roots = [
    r"L:\DL",
    r"L:\DL\datasets",
    r"L:\DL\datasets\shapenet",
    r"C:\Users",  # sometimes people put it here
]

found_obj = False
for root in search_roots:
    if not os.path.exists(root):
        continue
    # Look max 4 levels deep for .obj files
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.replace(root, "").count(os.sep)
        if depth > 5:
            dirnames.clear()
            continue
        for f in filenames:
            if f.endswith(".obj") and ("shapenet" in dirpath.lower() or "ShapeNet" in dirpath):
                print(f"  FOUND .obj: {os.path.join(dirpath, f)}")
                found_obj = True
                dirnames.clear()
                break
        if found_obj:
            break
    if found_obj:
        break

if not found_obj:
    print("  No ShapeNet .obj meshes found.")

# ── Check 2: Cap3D point cloud files ──────────────────
print("\n[2] Looking for Cap3D point cloud files (.npz, .pt, .npy)...")
cap3d_found = []
for root in search_roots:
    if not os.path.exists(root):
        continue
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.replace(root, "").count(os.sep)
        if depth > 5:
            dirnames.clear()
            continue
        for f in filenames:
            if "cap3d" in f.lower() or "cap3d" in dirpath.lower():
                if f.endswith((".npz", ".pt", ".npy", ".zip")):
                    full = os.path.join(dirpath, f)
                    size_mb = os.path.getsize(full) / 1e6
                    cap3d_found.append((full, size_mb))
                    if len(cap3d_found) >= 10:
                        dirnames.clear()
                        break
    if len(cap3d_found) >= 10:
        break

if cap3d_found:
    for path, size in cap3d_found:
        print(f"  {path} ({size:.1f} MB)")
else:
    print("  No Cap3D point cloud files found.")

# ── Check 3: What's in your datasets folder ───────────
print("\n[3] Contents of L:\\DL\\datasets\\shapenet\\")
shapenet_root = Path(r"L:\DL\datasets\shapenet")
if shapenet_root.exists():
    for item in sorted(shapenet_root.iterdir()):
        if item.is_dir():
            count = sum(1 for _ in item.iterdir())
            print(f"  [DIR]  {item.name}/  ({count} items)")
        else:
            size_mb = item.stat().st_size / 1e6
            print(f"  [FILE] {item.name}  ({size_mb:.1f} MB)")
else:
    print("  Directory not found")

# ── Check 4: Count Cap3D zips ─────────────────────────
print("\n[4] Cap3D render zips...")
zip_dir = Path(r"L:\DL\datasets\shapenet\renders\Cap3D_ShapeNet_renderimgs")
if zip_dir.exists():
    zips = list(zip_dir.glob("*.zip"))
    print(f"  Total zips: {len(zips)}")
    if zips:
        # Show synset distribution
        synsets = {}
        for z in zips:
            parts = z.stem.split("_", 1)
            if len(parts) == 2:
                synsets[parts[0]] = synsets.get(parts[0], 0) + 1
        print(f"  Categories: {len(synsets)}")
        for sid, count in sorted(synsets.items(), key=lambda x: -x[1])[:5]:
            print(f"    {sid}: {count} objects")
else:
    print("  Zip directory not found")

# ── Check 5: Disk space ──────────────────────────────
print("\n[5] Disk space...")
import shutil
for drive in ["L:\\", "C:\\"]:
    try:
        total, used, free = shutil.disk_usage(drive)
        print(f"  {drive} — Free: {free/1e9:.1f} GB / Total: {total/1e9:.1f} GB")
    except:
        pass

print("\n" + "=" * 60)
print("Copy everything above and paste it to me.")
