"""
Synchronous Extraction of all 12 tar.gz Archives to data/raw/images
-------------------------------------------------------------------
"""

import os
import glob
import tarfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
IMAGES_DIR = RAW_DIR / "images"
ARCHIVES_DIR = RAW_DIR / "archives"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def unpack_all_tarballs():
    print("=== STARTING SYNCHRONOUS TARBALL UNPACKING ===")
    t0 = time.time()
    
    archives = sorted(list(ARCHIVES_DIR.glob("images_*.tar.gz")))
    print(f"Found {len(archives)} tar.gz archives in {ARCHIVES_DIR}.")

    for idx, arch_path in enumerate(archives, 1):
        t_arch = time.time()
        print(f"[{idx}/{len(archives)}] Extracting {arch_path.name}...")
        with tarfile.open(arch_path, "r:gz") as tar:
            tar.extractall(RAW_DIR)
        elapsed_arch = time.time() - t_arch
        current_count = len(os.listdir(IMAGES_DIR))
        print(f"   Done {arch_path.name} in {elapsed_arch:.2f}s. Total PNGs in images/: {current_count}")

    final_count = len(os.listdir(IMAGES_DIR))
    print(f"\nAll archives extracted in {time.time() - t0:.2f} seconds.")
    print(f"FINAL VERIFIED PNG COUNT: {final_count} (Expected: 112,120)")
    return final_count == 112120


if __name__ == "__main__":
    unpack_all_tarballs()
