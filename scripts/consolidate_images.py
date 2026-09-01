"""
Consolidate and Extract All 112,120 PNG Images directly into Chest-Xray-AI/data/raw/images/
-----------------------------------------------------------------------------------------
"""

import os
import glob
import shutil
import tarfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # d:\XRAY-ABSTRACT\Chest-Xray-AI
TARGET_IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "images"
TARGET_ARCHIVES_DIR = PROJECT_ROOT / "data" / "raw" / "archives"

ALT_IMAGES_DIR = PROJECT_ROOT.parent / "data" / "raw" / "images"

TARGET_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def consolidate_and_extract():
    print(f"=== CONSOLIDATING IMAGES INTO {TARGET_IMAGES_DIR} ===")
    t0 = time.time()

    # 1. Move any images from alt directory if present
    if ALT_IMAGES_DIR.exists() and ALT_IMAGES_DIR.resolve() != TARGET_IMAGES_DIR.resolve():
        print(f"Moving PNGs from {ALT_IMAGES_DIR} -> {TARGET_IMAGES_DIR}...")
        moved_count = 0
        for f in ALT_IMAGES_DIR.glob("*.png"):
            dest = TARGET_IMAGES_DIR / f.name
            if not dest.exists():
                shutil.move(str(f), str(dest))
                moved_count += 1
        print(f"Moved {moved_count} images from alt directory.")
        try:
            shutil.rmtree(str(ALT_IMAGES_DIR.parent.parent))  # clean up alt root if empty
        except Exception:
            pass

    # 2. Flatten subfolders inside TARGET_IMAGES_DIR
    sub_img = TARGET_IMAGES_DIR / "images"
    if sub_img.exists():
        for f in sub_img.glob("*.png"):
            dest = TARGET_IMAGES_DIR / f.name
            if not dest.exists():
                shutil.move(str(f), str(dest))
        try:
            shutil.rmtree(str(sub_img))
        except Exception:
            pass

    # 3. Extract all 12 tarballs into TARGET_IMAGES_DIR
    archives = sorted(list(TARGET_ARCHIVES_DIR.glob("images_*.tar.gz")))
    print(f"Found {len(archives)} archives in {TARGET_ARCHIVES_DIR}.")

    for idx, arch in enumerate(archives, 1):
        print(f"[{idx}/{len(archives)}] Extracting {arch.name}...")
        with tarfile.open(arch, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.name.endswith(".png")]
            for m in members:
                fname = Path(m.name).name
                dest = TARGET_IMAGES_DIR / fname
                if not dest.exists():
                    f_in = tar.extractfile(m)
                    if f_in:
                        with open(dest, "wb") as f_out:
                            while chunk := f_in.read(4 * 1024 * 1024):
                                f_out.write(chunk)

    final_png_count = len(list(TARGET_IMAGES_DIR.glob("*.png")))
    print(f"\nConsolidation & Extraction Completed in {time.time() - t0:.2f} seconds.")
    print(f"Total PNGs in {TARGET_IMAGES_DIR}: {final_png_count} (Expected: 112,120)")
    return final_png_count == 112120


if __name__ == "__main__":
    consolidate_and_extract()
