"""
NIH ChestX-ray14 Zip & Tar Extraction Utility
---------------------------------------------
Unzips images-selected.zip (45 GB) into data/raw/archives/ and extracts all 112,120 PNG images into data/raw/images/.
"""

import os
import glob
import zipfile
import tarfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVES_DIR = PROJECT_ROOT / "data" / "raw" / "archives"
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "images"
ZIP_PATH = ARCHIVES_DIR / "images-selected.zip"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=== STARTING COMPLETE DATASET UNPACKING ===")
    start_time = time.time()

    if not ZIP_PATH.exists():
        print(f"ERROR: {ZIP_PATH} not found!")
        return

    print(f"Unzipping {ZIP_PATH} ({ZIP_PATH.stat().st_size / (1024**3):.2f} GB)...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        members = [m for m in zf.namelist() if m.endswith(".tar.gz")]
        print(f"Zip contains {len(members)} tar.gz archives.")
        for idx, member in enumerate(members, 1):
            target_path = ARCHIVES_DIR / Path(member).name
            if not target_path.exists() or target_path.stat().st_size == 0:
                print(f"[{idx}/{len(members)}] Unzipping {member} -> {target_path.name}...")
                with zf.open(member) as src, open(target_path, "wb") as dst:
                    shutil_copy(src, dst)
            else:
                print(f"[{idx}/{len(members)}] Already unzipped: {target_path.name}")

    tar_files = sorted(list(ARCHIVES_DIR.glob("images_*.tar.gz")))
    print(f"\nExtracting {len(tar_files)} tar.gz archives into data/raw/images/...")
    for idx, tar_path in enumerate(tar_files, 1):
        print(f"[{idx}/{len(tar_files)}] Extracting {tar_path.name}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".png"):
                    filename = Path(member.name).name
                    dest_file = IMAGES_DIR / filename
                    if not dest_file.exists():
                        f_src = tar.extractfile(member)
                        if f_src:
                            with open(dest_file, "wb") as f_dst:
                                shutil_copy(f_src, f_dst)

    total_pngs = len(list(IMAGES_DIR.glob("*.png")))
    elapsed = time.time() - start_time
    print(f"\nFinished unpacking in {elapsed:.2f} seconds.")
    print(f"Total PNGs in data/raw/images/: {total_pngs} (Expected: 112,120)")


def shutil_copy(src, dst, bufsize=4*1024*1024):
    while chunk := src.read(bufsize):
        dst.write(chunk)


if __name__ == "__main__":
    main()
