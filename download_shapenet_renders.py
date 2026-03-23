"""
=========================================================================================================
Cap3D ShapeNet Rendered Images Downloader
=========================================================================================================
Downloads rendered images for 52,472 ShapeNet objects from HuggingFace tiange/Cap3D
Each object has 8 rendered views with camera info (intrinsic + extrinsic), depth, and masks.

Source : https://huggingface.co/datasets/tiange/Cap3D/tree/main/RenderedImage_perobj_zips_ShapeNet
Target : L:\DL\datasets\shapenet\

Structure after extraction:
  L:\DL\datasets\shapenet\
    renders\
      <object_uid>.zip    (each contains 8 rendered images + camera info)
    images\               (extracted per-object folders with actual .png files)
      <object_uid>\
        000.png, 001.png, ..., 007.png   (8 rendered views)
        transforms.json                   (camera intrinsics + extrinsics)

Features:
  - Resume download if interrupted (checks existing files + partial .tmp downloads)
  - Progress bar with speed + ETA
  - Auto-extracts outer zip files to get per-object zips
  - Optional: extracts per-object zips into image folders

Author : Mrinal Bharadwaj
Project: AI 535 - Bridging Synthetic-to-Real Gap in 3D Reconstruction
=========================================================================================================
"""

import os
import sys
import time
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm


# ============================= CONFIGURATION =============================

BASE_DIR = Path(r"L:\DL\datasets\shapenet")
DOWNLOAD_DIR = BASE_DIR / "downloads"       # temporary zip storage
RENDERS_DIR  = BASE_DIR / "renders"         # per-object zip files after first extraction
IMAGES_DIR   = BASE_DIR / "images"          # final extracted images (optional)

# HuggingFace dataset base URL
HF_BASE_URL = "https://huggingface.co/datasets/tiange/Cap3D/resolve/main/RenderedImage_perobj_zips_ShapeNet"

# The ShapeNet rendered images folder has 3 large zip files
# Each outer zip contains many inner per-object zips
ZIP_FILES = [
    "compressed_imgs_perobj_00.zip",
    "compressed_imgs_perobj_01.zip",
    "compressed_imgs_perobj_02.zip",
]

CHUNK_SIZE = 8192  # 8KB chunks for download
EXTRACT_INNER_ZIPS = True  # Set to True to also extract per-object zips into image folders

# ShapeNet synset IDs -> category names (for organizing by category later)
SHAPENET_CATEGORIES = {
    "02691156": "airplane",
    "02747177": "ashcan",
    "02773838": "bag",
    "02801938": "basket",
    "02808440": "bathtub",
    "02818832": "bed",
    "02828884": "bench",
    "02843684": "birdhouse",
    "02871439": "bookshelf",
    "02876657": "bottle",
    "02880940": "bowl",
    "02924116": "bus",
    "02933112": "cabinet",
    "02942699": "camera",
    "02946921": "can",
    "02954340": "cap",
    "02958343": "car",
    "02992529": "cellphone",
    "03001627": "chair",
    "03046257": "clock",
    "03085013": "keyboard",
    "03207941": "dishwasher",
    "03211117": "display",
    "03261776": "earphone",
    "03325088": "faucet",
    "03337140": "file_cabinet",
    "03467517": "guitar",
    "03513137": "helmet",
    "03593526": "jar",
    "03624134": "knife",
    "03636649": "lamp",
    "03642806": "laptop",
    "03691459": "loudspeaker",
    "03710193": "mailbox",
    "03759954": "microphone",
    "03761084": "microwave",
    "03790512": "motorcycle",
    "03797390": "mug",
    "03928116": "piano",
    "03938244": "pillow",
    "03948459": "pistol",
    "03991062": "pot",
    "04004475": "printer",
    "04074963": "remote",
    "04090263": "rifle",
    "04099429": "rocket",
    "04225987": "skateboard",
    "04256520": "sofa",
    "04330267": "stove",
    "04379243": "table",
    "04401088": "telephone",
    "04460130": "tower",
    "04468005": "train",
    "04530566": "vessel",
    "04554684": "washer",
}


# ============================= HELPER FUNCTIONS =============================

def get_file_size_remote(url):
    """Get file size from server without downloading (HEAD request)"""
    try:
        response = requests.head(url, allow_redirects=True, timeout=30)
        if response.status_code == 200 and "Content-Length" in response.headers:
            return int(response.headers["Content-Length"])
    except requests.RequestException:
        pass
    return None


def format_size(size_bytes):
    """Convert bytes to human readable format"""
    if size_bytes is None:
        return "Unknown"
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def download_file(url, dest_path, description="Downloading"):
    """
    Download a file with resume support.
    If dest_path already exists and is complete => skip.
    If dest_path.tmp exists => resume from where it left off.
    """
    dest_path = Path(dest_path)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    # Check if already fully downloaded
    remote_size = get_file_size_remote(url)

    if dest_path.exists():
        local_size = dest_path.stat().st_size
        if remote_size is None or local_size >= remote_size:
            print(f"  [SKIP] {dest_path.name} already downloaded ({format_size(local_size)})")
            return True
        else:
            print(f"  [RESUME] {dest_path.name} is incomplete ({format_size(local_size)} / {format_size(remote_size)})")
            dest_path.rename(tmp_path)

    # Check for partial .tmp file to resume
    resume_from = 0
    if tmp_path.exists():
        resume_from = tmp_path.stat().st_size
        if remote_size and resume_from >= remote_size:
            tmp_path.rename(dest_path)
            print(f"  [DONE] {dest_path.name} was already complete")
            return True
        print(f"  [RESUME] Resuming {dest_path.name} from {format_size(resume_from)}")

    # Build request headers for resume
    headers = {}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=60)

        # If server doesn't support range requests, start over
        if resume_from > 0 and response.status_code == 200:
            print(f"  [WARNING] Server doesn't support resume. Re-downloading from start...")
            resume_from = 0
            tmp_path.unlink(missing_ok=True)
        elif response.status_code not in (200, 206):
            print(f"  [ERROR] HTTP {response.status_code} for {url}")
            return False

        # Total size for progress bar
        content_length = response.headers.get("Content-Length")
        total_size = int(content_length) + resume_from if content_length else remote_size

        mode = "ab" if resume_from > 0 else "wb"

        with open(tmp_path, mode) as f:
            with tqdm(
                total=total_size,
                initial=resume_from,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"  {description}",
                ncols=100,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        tmp_path.rename(dest_path)
        print(f"  [DONE] {dest_path.name} downloaded successfully ({format_size(dest_path.stat().st_size)})")
        return True

    except (requests.RequestException, ConnectionError, TimeoutError) as e:
        print(f"  [ERROR] Download interrupted: {e}")
        print(f"  [INFO] Partial file saved as {tmp_path.name} — run script again to resume")
        return False


def extract_outer_zip(zip_path, extract_to, description="Extracting"):
    """Extract outer zip (contains per-object inner zips). Skips if already done."""
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)

    marker = extract_to / f".extracted_{zip_path.stem}"
    if marker.exists():
        print(f"  [SKIP] {zip_path.name} already extracted")
        return True

    if not zip_path.exists():
        print(f"  [ERROR] Zip file not found: {zip_path}")
        return False

    print(f"  Extracting {zip_path.name} (outer zip → per-object zips)...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            for member in tqdm(members, desc=f"  Extracting", ncols=100):
                zf.extract(member, extract_to)

        marker.touch()
        print(f"  [DONE] Extracted {len(members)} inner zips from {zip_path.name}")
        return True

    except zipfile.BadZipFile:
        print(f"  [ERROR] Corrupted zip: {zip_path.name} — delete and re-run to re-download")
        return False


def extract_inner_zips(renders_dir, images_dir):
    """Extract each per-object zip into its own image folder."""
    renders_dir = Path(renders_dir)
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    inner_zips = sorted(renders_dir.rglob("*.zip"))
    if not inner_zips:
        print("  [WARNING] No per-object zip files found in renders directory")
        return

    print(f"  Found {len(inner_zips)} per-object zips to extract...")

    extracted_count = 0
    skipped_count = 0

    for zp in tqdm(inner_zips, desc="  Extracting per-object", ncols=100):
        obj_uid = zp.stem
        obj_dir = images_dir / obj_uid

        # Skip if already extracted (folder exists and has files)
        if obj_dir.exists() and any(obj_dir.iterdir()):
            skipped_count += 1
            continue

        try:
            obj_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zp, "r") as zf:
                zf.extractall(obj_dir)
            extracted_count += 1
        except (zipfile.BadZipFile, Exception) as e:
            # Skip corrupted per-object zips silently
            pass

    print(f"  [DONE] Extracted: {extracted_count}, Skipped (already done): {skipped_count}")


# ============================= MAIN DOWNLOAD LOGIC =============================

def main():
    print("=" * 90)
    print("  Cap3D ShapeNet Rendered Images Downloader")
    print("  Source : huggingface.co/datasets/tiange/Cap3D/RenderedImage_perobj_zips_ShapeNet")
    print(f"  Target : {BASE_DIR}")
    print("  Images : 8 rendered views per object + camera info (intrinsic & extrinsic)")
    print("=" * 90)
    print()

    # Create directories
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[1/4] Directories ready")
    print(f"       Downloads : {DOWNLOAD_DIR}")
    print(f"       Renders   : {RENDERS_DIR}")
    print(f"       Images    : {IMAGES_DIR}")
    print()

    # Download outer zip files
    print(f"[2/4] Downloading rendered image zip files ({len(ZIP_FILES)} files)...")
    print(f"       NOTE: These are LARGE files (several GB each). Be patient.")
    all_downloaded = True
    for i, zip_name in enumerate(ZIP_FILES):
        url = f"{HF_BASE_URL}/{zip_name}"
        dest = DOWNLOAD_DIR / zip_name
        print(f"\n  --- File {i + 1}/{len(ZIP_FILES)}: {zip_name} ---")
        success = download_file(url, dest, description=zip_name)
        if not success:
            all_downloaded = False

    if not all_downloaded:
        print("\n[WARNING] Some downloads failed. Run this script again to resume.")
        print("          Partial downloads are saved and will be resumed automatically.")
    print()

    # Extract outer zips (each outer zip → many per-object inner zips)
    print(f"[3/4] Extracting outer zip files → per-object zips...")
    for zip_name in ZIP_FILES:
        zip_path = DOWNLOAD_DIR / zip_name
        if zip_path.exists():
            extract_outer_zip(zip_path, RENDERS_DIR)
    print()

    # Optionally extract inner per-object zips
    if EXTRACT_INNER_ZIPS:
        print(f"[4/4] Extracting per-object zips → image folders...")
        extract_inner_zips(RENDERS_DIR, IMAGES_DIR)
    else:
        print(f"[4/4] Skipping per-object extraction (set EXTRACT_INNER_ZIPS = True to enable)")
    print()

    # Final verification
    inner_zips = list(RENDERS_DIR.rglob("*.zip"))
    image_folders = [d for d in IMAGES_DIR.iterdir() if d.is_dir()] if IMAGES_DIR.exists() else []

    print("=" * 90)
    print(f"  SUMMARY")
    print(f"  Per-object zip files    : {len(inner_zips)}")
    print(f"  Extracted image folders : {len(image_folders)}")
    print(f"  Views per object        : 8 rendered images")
    print(f"  Location                : {BASE_DIR}")
    print()
    print(f"  ShapeNet Category Map (for reference):")
    # Print a few key categories
    key_cats = ["02691156", "02958343", "03001627", "04256520", "04379243"]
    for synset_id in key_cats:
        if synset_id in SHAPENET_CATEGORIES:
            print(f"    {synset_id} => {SHAPENET_CATEGORIES[synset_id]}")
    print(f"    ... and {len(SHAPENET_CATEGORIES) - len(key_cats)} more categories")
    print("=" * 90)

    if len(inner_zips) == 0 and len(image_folders) == 0:
        print("\n[WARNING] No data found! Check if downloads completed properly.")
    else:
        print(f"\n[SUCCESS] ShapeNet rendered images ready!")
        print(f"          Ready for training pipeline.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
