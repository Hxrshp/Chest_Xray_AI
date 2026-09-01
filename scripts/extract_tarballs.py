"""
Extract all 12 tar.gz archives into data/raw/images
---------------------------------------------------
"""

import os
import glob
import tarfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVES_DIR = PROJECT_ROOT / "data" / "raw" / "archives"
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "images"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def extract_tarballs():
    print("=== EXTRACTING TARBALLS INTO DATA/RAW/IMAGES ===")
    t0 = time.time()
    tar_files = sorted(list(ARCHIVES_DIR.glob("images_*.tar.gz")))
    print(f"Found {len(tar_files)} tar.gz archives.")

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
                                f_dst.write(f_src.read())

    final_png_count = len(list(IMAGES_DIR.glob("*.png")))
    print(f"\nFinished extracting tarballs in {time.time() - t0:.2f} seconds.")
    print(f"Total PNGs in data/raw/images/: {final_png_count} (Expected: 112,120)")


if __name__ == "__main__":
    extract_tarballs()
