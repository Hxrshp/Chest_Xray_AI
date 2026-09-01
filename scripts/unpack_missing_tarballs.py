"""
Unpack Missing Tarballs into data/raw/images/
---------------------------------------------
Robustly extracts any unextracted PNG files from all 12 tarballs.
"""

import os
import tarfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
IMAGES_DIR = RAW_DIR / "images"
ARCHIVES_DIR = RAW_DIR / "archives"


def unpack_missing():
    print("=== STARTING UNPACKING OF MISSING IMAGES ===")
    t0 = time.time()
    
    tar_files = sorted(list(ARCHIVES_DIR.glob("images_*.tar.gz")))
    print(f"Found {len(tar_files)} tar.gz archives in {ARCHIVES_DIR}.")

    extracted_new = 0

    for idx, arch_path in enumerate(tar_files, 1):
        print(f"[{idx}/{len(tar_files)}] Checking {arch_path.name}...")
        t_arch = time.time()
        count_before = len(os.listdir(IMAGES_DIR))
        try:
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
                                extracted_new += 1
        except Exception as e:
            print(f"  WARNING: Archive {arch_path.name} read error: {e}")

        count_after = len(os.listdir(IMAGES_DIR))
        print(f"   Done {arch_path.name} in {time.time() - t_arch:.1f}s. Added: {count_after - count_before}, Current total: {count_after}")

    final_count = len(os.listdir(IMAGES_DIR))
    print(f"\nUnpacking complete in {time.time() - t0:.2f} seconds.")
    print(f"TOTAL PNG COUNT IN {IMAGES_DIR}: {final_count} (Expected: 112,120)")
    return final_count == 112120


if __name__ == "__main__":
    unpack_missing()
