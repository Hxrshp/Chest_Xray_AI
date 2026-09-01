"""
Unpack Full Dataset (112,120 PNG images) into data/raw/images/
--------------------------------------------------------------
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


def unpack_full_dataset():
    print("=== STARTING FULL DATASET UNPACKING ===")
    t0 = time.time()
    
    tar_files = sorted(list(ARCHIVES_DIR.glob("images_*.tar.gz")))
    print(f"Found {len(tar_files)} tar.gz archives in {ARCHIVES_DIR}.")

    for idx, arch_path in enumerate(tar_files, 1):
        print(f"[{idx}/{len(tar_files)}] Unpacking {arch_path.name}...")
        t_arch = time.time()
        with tarfile.open(arch_path, "r:gz") as tar:
            for member in tar:
                if member.isfile() and member.name.endswith(".png"):
                    fname = os.path.basename(member.name)
                    dest_path = IMAGES_DIR / fname
                    if not dest_path.exists():
                        f_in = tar.extractfile(member)
                        if f_in:
                            with open(dest_path, "wb") as f_out:
                                while chunk := f_in.read(4 * 1024 * 1024):
                                    f_out.write(chunk)
        elapsed = time.time() - t_arch
        curr_count = len(os.listdir(IMAGES_DIR))
        print(f"   Done {arch_path.name} in {elapsed:.1f}s. Current PNG count: {curr_count}")

    final_count = len(os.listdir(IMAGES_DIR))
    print(f"\nUnpacking finished in {time.time() - t0:.2f} seconds.")
    print(f"TOTAL PNG COUNT IN {IMAGES_DIR}: {final_count} (Expected: 112,120)")
    return final_count == 112120


if __name__ == "__main__":
    unpack_full_dataset()
