"""
Extract Remaining NIH ChestX-ray14 Tarballs (003 through 012) into data/raw/images
----------------------------------------------------------------------------------
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


def extract_remaining():
    print("=== EXTRACTING REMAINING IMAGE ARCHIVES (003 THROUGH 012) ===")
    t0 = time.time()
    
    tar_files = sorted(list(ARCHIVES_DIR.glob("images_*.tar.gz")))
    print(f"Found {len(tar_files)} total tarball archives.")

    for idx, tar_path in enumerate(tar_files, 1):
        print(f"[{idx}/{len(tar_files)}] Opening {tar_path.name}...")
        extracted_this_tar = 0
        with tarfile.open(tar_path, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.name.endswith(".png")]
            print(f"   Found {len(members)} PNG members in {tar_path.name}.")
            for m in members:
                fname = Path(m.name).name
                dest_path = IMAGES_DIR / fname
                if not dest_path.exists():
                    f_in = tar.extractfile(m)
                    if f_in:
                        with open(dest_path, "wb") as f_out:
                            shutil_copy(f_in, f_out)
                        extracted_this_tar += 1
        print(f"   Extracted {extracted_this_tar} new images from {tar_path.name}.")

    final_png_count = len(list(IMAGES_DIR.glob("*.png")))
    print(f"\nExtraction completed in {time.time() - t0:.2f} seconds.")
    print(f"Total PNGs in data/raw/images/: {final_png_count} (Expected: 112,120)")


def shutil_copy(src, dst, bufsize=8*1024*1024):
    while chunk := src.read(bufsize):
        dst.write(chunk)


if __name__ == "__main__":
    extract_remaining()
