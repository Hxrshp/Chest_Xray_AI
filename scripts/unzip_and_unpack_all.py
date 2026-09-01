"""
NIH ChestX-ray14 Robust Zip & Tar Unpacking Script
--------------------------------------------------
1. Verifies/unzips all 12 tar.gz archives from data/raw/archives/images-selected.zip.
2. Flattens and extracts all 112,120 PNG images directly into data/raw/images/.
3. Removes subfolder artifacts.
"""

import os
import shutil
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


def run_unpacking():
    print("=== STARTING ROBUST DATASET UNPACKING ===")
    t0 = time.time()

    # Move any PNGs from subfolders into IMAGES_DIR
    sub_images_dir = IMAGES_DIR / "images"
    if sub_images_dir.exists():
        for child in sub_images_dir.glob("*.png"):
            target_dest = IMAGES_DIR / child.name
            if not target_dest.exists():
                shutil.move(str(child), str(target_dest))
        try:
            shutil.rmtree(str(sub_images_dir))
        except Exception:
            pass

    # Step 1: Ensure all 12 tarballs are extracted from images-selected.zip
    if ZIP_PATH.exists():
        print(f"Checking {ZIP_PATH.name} ({ZIP_PATH.stat().st_size / (1024**3):.2f} GB)...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            for member_name in zf.namelist():
                if member_name.endswith(".tar.gz"):
                    basename = Path(member_name).name
                    target_file = ARCHIVES_DIR / basename
                    expected_sz = EXPECTED_TAR_SIZES.get(basename, 0)
                    
                    if not target_file.exists() or target_file.stat().st_size != expected_sz:
                        print(f"Unzipping {basename} from zip archive...")
                        with zf.open(member_name) as src, open(target_file, "wb") as dst:
                            while chunk := src.read(8 * 1024 * 1024):
                                dst.write(chunk)
                        print(f"  Saved {basename} ({target_file.stat().st_size} bytes)")
                    else:
                        print(f"  Verified archive present: {basename}")

    # Step 2: Extract all 12 tarballs into IMAGES_DIR
    print("\n--- Extracting PNG images from all 12 tarballs into data/raw/images/ ---")
    for tar_name in sorted(EXPECTED_TAR_SIZES.keys()):
        tar_path = ARCHIVES_DIR / tar_name
        if not tar_path.exists():
            print(f"CRITICAL ERROR: {tar_name} missing after zip extraction!")
            continue

        print(f"Extracting PNGs from {tar_name}...")
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
    print(f"\nUnpacking finished in {time.time() - t0:.2f} seconds.")
    print(f"Total PNGs in data/raw/images/: {final_png_count} (Expected: 112,120)")
    return final_png_count == 112120


if __name__ == "__main__":
    run_unpacking()
