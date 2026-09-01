"""
Simple Robust PNG Extraction Script
------------------------------------
Extracts all PNG files from data/raw/archives/images_*.tar.gz into data/raw/images/.
"""

import os
import glob
import tarfile
import time

RAW_IMAGES_DIR = "data/raw/images"
ARCHIVES_DIR = "data/raw/archives"

os.makedirs(RAW_IMAGES_DIR, exist_ok=True)


def extract_all():
    print("=== STARTING ROBUST PNG EXTRACTION ===")
    t0 = time.time()
    archives = sorted(glob.glob(os.path.join(ARCHIVES_DIR, "images_*.tar.gz")))
    print(f"Found {len(archives)} archive files.")

    total_extracted = 0
    for idx, arch in enumerate(archives, 1):
        print(f"[{idx}/{len(archives)}] Processing {os.path.basename(arch)}...")
        extracted_this_archive = 0
        with tarfile.open(arch, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".png"):
                    fname = os.path.basename(member.name)
                    dest_path = os.path.join(RAW_IMAGES_DIR, fname)
                    if not os.path.exists(dest_path):
                        f_in = tar.extractfile(member)
                        if f_in:
                            with open(dest_path, "wb") as f_out:
                                f_out.write(f_in.read())
                            extracted_this_archive += 1
        print(f"   Extracted {extracted_this_archive} new PNGs from {os.path.basename(arch)}.")
        total_extracted += extracted_this_archive

    final_count = len(glob.glob(os.path.join(RAW_IMAGES_DIR, "*.png")))
    print(f"\nExtraction Finished in {time.time() - t0:.2f} seconds.")
    print(f"Total PNGs in {RAW_IMAGES_DIR}: {final_count} (Expected: 112,120)")


if __name__ == "__main__":
    extract_all()
