"""
Flattened Archive Extraction Script
------------------------------------
Extracts every PNG from all 12 tarballs directly into data/raw/images/ filename.png.
"""

import os
import glob
import tarfile
import time

TARGET_DIR = "data/raw/images"
ARCHIVES_DIR = "data/raw/archives"
os.makedirs(TARGET_DIR, exist_ok=True)


def main():
    print("=== STARTING DIRECT FLATTENED EXTRACTION ===")
    t0 = time.time()
    archives = sorted(glob.glob(os.path.join(ARCHIVES_DIR, "images_*.tar.gz")))
    print(f"Found {len(archives)} archives.")

    for idx, arch in enumerate(archives, 1):
        print(f"[{idx}/{len(archives)}] Processing {os.path.basename(arch)}...")
        with tarfile.open(arch, "r:gz") as tar:
            for m in tar.getmembers():
                if m.name.endswith(".png"):
                    dest = os.path.join(TARGET_DIR, os.path.basename(m.name))
                    if not os.path.exists(dest):
                        f_in = tar.extractfile(m)
                        if f_in:
                            with open(dest, "wb") as f_out:
                                f_out.write(f_in.read())

    final_count = len(os.listdir(TARGET_DIR))
    print(f"\nExtraction completed in {time.time() - t0:.2f} seconds.")
    print(f"Total count in {TARGET_DIR}: {final_count} (Expected: 112,120)")


if __name__ == "__main__":
    main()
