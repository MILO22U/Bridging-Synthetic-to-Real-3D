"""
=========================================================================================================
Cap3D ShapeNet Point Cloud Downloader
=========================================================================================================
Downloads 52,472 ShapeNet point clouds (16,384 colored points each) from HuggingFace tiange/Cap3D
Each .ply file has shape [16384, 6] => (x, y, z, r, g, b)

Source : https://huggingface.co/datasets/tiange/Cap3D/tree/main/PointCloud_zips_ShapeNet
Target : L:\DL\datasets\cap3d\

Features:
  - Resume download if interrupted (checks existing files + partial .tmp downloads)
  - Progress bar with speed + ETA
  - Auto-extracts zip files after download
  - Verifies .ply file count after extraction

Author : Mrinal Bharadwaj
Project: AI 535 - Bridging Synthetic-to-Real Gap in 3D Reconstruction
=========================================================================================================
"""

import os
import sys
import time
import zipfile
import hashlib
import requests
from pathlib import Path
from tqdm import tqdm


# ============================= CONFIGURATION =============================

BASE_DIR = Path(r"L:\DL\datasets\cap3d")
DOWNLOAD_DIR = BASE_DIR / "downloads"       # temporary zip storage
EXTRACT_DIR  = BASE_DIR / "pointclouds"     # extracted .ply files go here

# HuggingFace dataset base URL for Cap3D ShapeNet point clouds
HF_BASE_URL = "https://huggingface.co/datasets/tiange/Cap3D/resolve/main/PointCloud_zips_ShapeNet"

# The ShapeNet point cloud folder has 2 zip files (compressed_pcs_00.zip and compressed_pcs_01.zip)
# Each zip contains thousands of .ply files named by ShapeNet object UID
ZIP_FILES = [
    "compressed_pcs_00.zip",
    "compressed_pcs_01.zip",
]

# Also download the ShapeNet captions CSV for object metadata
CAPTION_URL = "https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/Cap3D_automated_ShapeNet.csv"
CAPTION_FILE = "Cap3D_automated_ShapeNet.csv"

CHUNK_SIZE = 8192  # 8KB chunks for download
EXPECTED_TOTAL_OBJECTS = 52472  # approximate number of ShapeNet objects


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
            # Incomplete file without .tmp extension — rename to .tmp for resume
            print(f"  [RESUME] {dest_path.name} is incomplete ({format_size(local_size)} / {format_size(remote_size)})")
            dest_path.rename(tmp_path)

    # Check for partial .tmp file to resume
    resume_from = 0
    if tmp_path.exists():
        resume_from = tmp_path.stat().st_size
        if remote_size and resume_from >= remote_size:
            # .tmp is actually complete, just rename
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

        # Open file in append mode if resuming, write mode otherwise
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

        # Rename .tmp to final filename on successful completion
        tmp_path.rename(dest_path)
        print(f"  [DONE] {dest_path.name} downloaded successfully ({format_size(dest_path.stat().st_size)})")
        return True

    except (requests.RequestException, ConnectionError, TimeoutError) as e:
        print(f"  [ERROR] Download interrupted: {e}")
        print(f"  [INFO] Partial file saved as {tmp_path.name} — run script again to resume")
        return False


def extract_zip(zip_path, extract_to, description="Extracting"):
    """Extract a zip file with progress tracking. Skips if already extracted."""
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)

    if not zip_path.exists():
        print(f"  [ERROR] Zip file not found: {zip_path}")
        return False

    # Check if already extracted by looking for a marker file
    marker = extract_to / f".extracted_{zip_path.stem}"
    if marker.exists():
        print(f"  [SKIP] {zip_path.name} already extracted")
        return True

    print(f"  {description} {zip_path.name}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            for member in tqdm(members, desc=f"  Extracting", ncols=100):
                zf.extract(member, extract_to)

        # Write marker file
        marker.touch()
        print(f"  [DONE] Extracted {len(members)} files from {zip_path.name}")
        return True

    except zipfile.BadZipFile:
        print(f"  [ERROR] Corrupted zip: {zip_path.name} — delete it and re-run to re-download")
        return False


# ============================= MAIN DOWNLOAD LOGIC =============================

def main():
    print("=" * 90)
    print("  Cap3D ShapeNet Point Cloud Downloader")
    print("  Source : huggingface.co/datasets/tiange/Cap3D/PointCloud_zips_ShapeNet")
    print(f"  Target : {BASE_DIR}")
    print("=" * 90)
    print()

    # Create directories
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[1/4] Directories ready")
    print(f"       Downloads : {DOWNLOAD_DIR}")
    print(f"       Extracted : {EXTRACT_DIR}")
    print()

    # Download captions CSV
    print(f"[2/4] Downloading ShapeNet captions CSV...")
    caption_dest = BASE_DIR / CAPTION_FILE
    download_file(CAPTION_URL, caption_dest, description=CAPTION_FILE)
    print()

    # Download point cloud zip files
    print(f"[3/4] Downloading point cloud zip files ({len(ZIP_FILES)} files)...")
    all_downloaded = True
    for i, zip_name in enumerate(ZIP_FILES):
        url = f"{HF_BASE_URL}/{zip_name}"
        dest = DOWNLOAD_DIR / zip_name
        print(f"\n  --- File {i + 1}/{len(ZIP_FILES)}: {zip_name} ---")
        success = download_file(url, dest, description=zip_name)
        if not success:
            all_downloaded = False
            print(f"  [WARNING] {zip_name} failed — run script again to retry/resume")

    if not all_downloaded:
        print("\n[WARNING] Some downloads failed. Run this script again to resume.")
        print("          Partial downloads are saved and will be resumed automatically.")
    print()

    # Extract zip files
    print(f"[4/4] Extracting point cloud .ply files...")
    for zip_name in ZIP_FILES:
        zip_path = DOWNLOAD_DIR / zip_name
        if zip_path.exists():
            extract_zip(zip_path, EXTRACT_DIR)
    print()

    # Final verification — count .ply files
    ply_files = list(EXTRACT_DIR.rglob("*.ply"))
    print("=" * 90)
    print(f"  SUMMARY")
    print(f"  Point cloud .ply files found : {len(ply_files)}")
    print(f"  Expected (approximate)       : ~{EXPECTED_TOTAL_OBJECTS}")
    print(f"  Location                     : {EXTRACT_DIR}")
    print(f"  Each file shape              : [16384, 6] => (x, y, z, r, g, b)")
    print("=" * 90)

    if len(ply_files) == 0:
        print("\n[WARNING] No .ply files found! Check if downloads completed properly.")
    elif len(ply_files) < EXPECTED_TOTAL_OBJECTS * 0.9:
        print(f"\n[WARNING] Found fewer files than expected. Some zips may have failed.")
    else:
        print(f"\n[SUCCESS] All point clouds downloaded and extracted!")
        print(f"          Ready for training pipeline.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
