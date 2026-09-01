"""
NIH ChestX-ray14 Archive Unpacking Script
------------------------------------------
Unzips images-selected.zip into data/raw/archives/ (images_001.tar.gz .. images_012.tar.gz)
and extracts all 112,120 PNG images into data/raw/images/.
"""

import os
import glob
import zipfile
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVES_DIR = PROJECT_ROOT / "data" / "raw" / "archives"
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "images"
ZIP_PATH = ARCHIVES_DIR / "images-selected.zip"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def extract_all():
    print("=== UNPACKING NIH CHESTX-RAY14 ARCHIVES ===")

    # 1. Unzip images-selected.zip if tar.gz files are missing
    if ZIP_PATH.exists():
        print(f"Opening {ZIP_PATH} ({ZIP_PATH.stat().st_size / (1024**3):.2f} GB)...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            for member in zf.namelist():
                if member.endswith(".tar.gz"):
                    target_file = ARCHIVES_DIR / Path(member).name
                    if not target_file.exists() or target_file.stat().st_size == 0:
                        print(f"Extracting {member} -> {target_file}...")
                        with zf.open(member) as src, open(target_file, "wb") as dst:
                            while chunk := src.read(4 * 1024 * 1024):
                                dst.write(chunk)

    # 2. Extract each tar.gz into data/raw/images/
    tar_files = sorted(list(ARCHIVES_DIR.glob("images_*.tar.gz")))
    print(f"Found {len(tar_files)} tar.gz archives.")

    for tar_path in tar_files:
        print(f"Extracting PNGs from {tar_path.name}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".png"):
                    member.name = Path(member.name).name  # Flatten path to images/ filename.png
                    tar.extract(member, path=IMAGES_DIR)

    final_png_count = len(list(IMAGES_DIR.glob("*.png")))
    print(f"\nExtraction Complete! Total PNGs in data/raw/images/: {final_png_count} (Expected: 112,120)")
    return final_png_count == 112120


if __name__ == "__main__":
    extract_all()
