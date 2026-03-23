"""
Quick diagnostic: what data do you actually have?
Run this and share the output with me.

Usage: python check_data.py --data-root ./data
"""

import os
import sys
from pathlib import Path

DATA_ROOT = sys.argv[1] if len(sys.argv) > 1 else "./data"

print(f"Checking: {os.path.abspath(DATA_ROOT)}")
print("=" * 60)

# 1. Check ShapeNet renderings
render_dir = Path(DATA_ROOT) / "shapenet" / "ShapeNetRendering"
render_dir2 = Path(DATA_ROOT) / "shapenet" / "renders"

for label, d in [("ShapeNetRendering", render_dir), ("renders", render_dir2)]:
    if d.exists():
        synsets = [s for s in d.iterdir() if s.is_dir()]
        total = sum(1 for s in synsets for m in s.iterdir() if m.is_dir())
        print(f"\n[Renders] {d}")
        print(f"  Categories: {len(synsets)}")
        print(f"  Total models: {total}")
        # Show a sample model to see file naming
        for s in synsets[:1]:
            for m in list(s.iterdir())[:1]:
                if m.is_dir():
                    files = list(m.rglob("*.png"))
                    print(f"  Sample: {m}")
                    print(f"    PNG files: {len(files)}")
                    if files:
                        print(f"    First: {files[0].relative_to(d)}")
                        print(f"    Last:  {files[-1].relative_to(d)}")
    else:
        print(f"\n[Renders] {d} — NOT FOUND")

# 2. Check for ShapeNet meshes (ShapeNetCore.v2)
for mesh_root in [
    Path(DATA_ROOT) / "shapenet",
    Path(DATA_ROOT) / "ShapeNetCore.v2",
    Path(DATA_ROOT) / ".." / "ShapeNetCore.v2",
    Path(DATA_ROOT) / "shapenet" / "ShapeNetCore.v2",
]:
    if not mesh_root.exists():
        continue
    # Look for .obj files
    objs = list(mesh_root.rglob("*.obj"))[:5]
    if objs:
        print(f"\n[Meshes] Found .obj files under {mesh_root}")
        print(f"  Sample: {objs[0]}")
        print(f"  Total .obj (sampled): {len(list(mesh_root.rglob('*.obj')))}")
        break
else:
    print(f"\n[Meshes] No .obj files found near {DATA_ROOT}")

# 3. Check Cap3D
cap3d_dir = Path(DATA_ROOT) / "cap3d"
if cap3d_dir.exists():
    print(f"\n[Cap3D] {cap3d_dir}")
    for f in cap3d_dir.rglob("*"):
        if f.is_file():
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.relative_to(cap3d_dir)} — {size_mb:.1f} MB")
    npy_count = len(list((cap3d_dir / "point_clouds").glob("*.npy"))) if (cap3d_dir / "point_clouds").exists() else 0
    print(f"  .npy files: {npy_count}")
else:
    print(f"\n[Cap3D] {cap3d_dir} — NOT FOUND")

# 4. Check disk space
try:
    import shutil
    total, used, free = shutil.disk_usage(DATA_ROOT)
    print(f"\n[Disk] Free: {free/1e9:.1f} GB")
except:
    pass

print("\n" + "=" * 60)
print("Copy everything above and share it with me.")
