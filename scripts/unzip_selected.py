"""
Extract all tar.gz archives from images-selected.zip into data/raw/archives/
-----------------------------------------------------------------------------
"""

import os
import zipfile
import tarfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVES_DIR = PROJECT_ROOT / "data" / "raw" / "archives"
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "images"
ZIP_PATH = ARCHIVES_DIR / "images-selected.zip"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_TAR_SIZES = {
    "images_001.tar.gz": 2008470987,
    "images_002.tar.gz": 3952623504,
    "images_003.tar.gz": 3929234850,
    "images_004.tar.gz": 3838903983,
    "images_005.tar.gz": 3935496531,
    "images_006.tar.gz": 3986301172,
    "images_007.tar.gz": 4016328426,
    "images_008.tar.gz": 4018347353,
    "images_009.tar.gz": 4111327929,
    "images_010.tar.gz": 4181556296,
    "images_011.tar.gz": 4187084020,
    "images_012.tar.gz": 2914187733,
}


def extract_zip_and_tars():
    print("=== UNPACKING ALL 12 CHESTX-RAY14 TARBALLS ===")
    t0 = time.time()

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        namelist = zf.namelist()
        print(f"Zip contains {len(namelist)} items.")
        
        for name in sorted(namelist):
            if name.endswith(".tar.gz"):
                basename = Path(name).name
                target_tar = ARCHIVES_DIR / basename
                expected_sz = EXPECTED_TAR_SIZES.get(basename, 0)
                
                if not target_tar.exists() or target_tar.stat().st_size != expected_sz:
                    print(f"Unzipping {basename} from zip archive...")
                    with zf.open(name) as src, open(target_tar, "wb") as dst:
                        while chunk := src.read(8 * 1024 * 1024):
                            dst.write(chunk)
                    print(f"  Saved {basename} ({target_tar.stat().st_size} bytes)")
                else:
                    print(f"Verified archive present: {basename}")

    print("\n--- Extracting PNG images from tarballs into data/raw/images/ ---")
    for tar_name in sorted(EXPECTED_TAR_SIZES.keys()):
        tar_path = ARCHIVES_DIR / tar_name
        print(f"Extracting {tar_name}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".png"):
                    fname = Path(member.name).name
                    dest_png = IMAGES_DIR / fname
                    if not dest_png.exists():
                        f_src = tar.extractfile(member)
                        if f_src:
                            with open(dest_png, "wb") as f_dst:
                                f_dst.write(f_src.read())

    final_png_count = len(list(IMAGES_DIR.glob("*.png")))
    print(f"\nUnpacking completed in {time.time() - t0:.2f} seconds.")
    print(f"Total PNGs in data/raw/images/: {final_png_count} (Expected: 112,120)")


if __name__ == "__main__":
    extract_zip_and_tars()
