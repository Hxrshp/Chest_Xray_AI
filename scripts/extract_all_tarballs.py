"""
Extract All 12 Tarballs into data/raw/images/
---------------------------------------------
"""

import os
import tarfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
IMAGES_DIR = RAW_DIR / "images"
ARCHIVES_DIR = RAW_DIR / "archives"


def extract_all():
    print("=== EXTRACTING ALL 12 TARBALLS TO DATA/RAW/IMAGES ===")
    t0 = time.time()
    
    tar_files = sorted(list(ARCHIVES_DIR.glob("images_*.tar.gz")))
    print(f"Found {len(tar_files)} tarball archives.")

    existing = set(os.listdir(IMAGES_DIR))
    print(f"Current existing PNG count: {len(existing):,}")

    for idx, arch in enumerate(tar_files, 1):
        print(f"[{idx}/12] Processing {arch.name}...")
        t_arch = time.time()
        added = 0
        with tarfile.open(arch, "r:gz") as tar:
            for member in tar:
                if member.isfile() and member.name.endswith(".png"):
                    fname = os.path.basename(member.name)
                    if fname not in existing:
                        f_in = tar.extractfile(member)
                        if f_in:
                            dest_path = IMAGES_DIR / fname
                            with open(dest_path, "wb") as f_out:
                                shutil_copy(f_in, f_out)
                            existing.add(fname)
                            added += 1
        print(f"   Done {arch.name} in {time.time() - t_arch:.1f}s. Added: {added:,}, Total now: {len(existing):,}")

    final_cnt = len(os.listdir(IMAGES_DIR))
    print(f"\nExtraction complete in {time.time() - t0:.1f}s. Final count: {final_cnt:,} (Expected: 112,120)")
    return final_cnt == 112120


def shutil_copy(f_in, f_out):
    while chunk := f_in.read(4 * 1024 * 1024):
        f_out.write(chunk)


if __name__ == "__main__":
    extract_all()
