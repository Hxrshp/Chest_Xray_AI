"""
Correct Stream-based Extraction of NIH ChestX-ray14 Tarballs
-------------------------------------------------------------
Uses for member in tar: iterator to guarantee sequential extraction without seeking failures.
"""

import os
import glob
import shutil
import tarfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVES_DIR = PROJECT_ROOT / "data" / "raw" / "archives"
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "images"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def extract_correct():
    print("=== STARTING SEQUENTIAL STREAM EXTRACTION ===")
    t0 = time.time()
    
    tar_files = sorted(list(ARCHIVES_DIR.glob("images_*.tar.gz")))
    print(f"Found {len(tar_files)} tarball archives.")

    total_extracted = 0
    for idx, tar_path in enumerate(tar_files, 1):
        print(f"[{idx}/{len(tar_files)}] Unpacking {tar_path.name}...")
        extracted_this_tar = 0
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar:
                if member.isfile() and member.name.endswith(".png"):
                    fname = os.path.basename(member.name)
                    dest_path = IMAGES_DIR / fname
                    if not dest_path.exists():
                        f_in = tar.extractfile(member)
                        if f_in:
                            with open(dest_path, "wb") as f_out:
                                shutil.copyfileobj(f_in, f_out, length=4*1024*1024)
                            extracted_this_tar += 1
        print(f"   Finished {tar_path.name} ({extracted_this_tar} new images extracted).")
        total_extracted += extracted_this_tar

    final_count = sum(1 for e in os.scandir(IMAGES_DIR) if e.name.endswith(".png"))
    print(f"\nStream extraction complete in {time.time() - t0:.2f} seconds.")
    print(f"Total PNGs in {IMAGES_DIR}: {final_count} (Expected: 112,120)")
    return final_count == 112120


if __name__ == "__main__":
    extract_correct()
