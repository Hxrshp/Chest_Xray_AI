"""
Fast Parallel Extraction of 12 tar.gz Archives to data/raw/images
-----------------------------------------------------------------
Uses concurrent futures to unpack all 12 tarball archives into data/raw/images/.
"""

import os
import tarfile
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVES_DIR = PROJECT_ROOT / "data" / "raw" / "archives"
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "images"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def extract_single_tarball(tar_path):
    print(f"Extracting {tar_path.name}...")
    extracted_count = 0
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".png"):
                filename = Path(member.name).name
                dest_file = IMAGES_DIR / filename
                if not dest_file.exists():
                    f_src = tar.extractfile(member)
                    if f_src:
                        with open(dest_file, "wb") as f_dst:
                            f_dst.write(f_src.read())
                        extracted_count += 1
    print(f"Finished {tar_path.name} ({extracted_count} new images extracted).")
    return tar_path.name, extracted_count


def main():
    print("=== FAST PARALLEL TARBALL EXTRACTION ===")
    t0 = time.time()
    tar_files = sorted(list(ARCHIVES_DIR.glob("images_*.tar.gz")))
    print(f"Found {len(tar_files)} tar.gz archives.")

    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(extract_single_tarball, p) for p in tar_files]
        for future in as_completed(futures):
            name, count = future.result()

    total_pngs = len(list(IMAGES_DIR.glob("*.png")))
    print(f"\nParallel extraction finished in {time.time() - t0:.2f} seconds.")
    print(f"Total PNGs in data/raw/images/: {total_pngs} (Expected: 112,120)")


if __name__ == "__main__":
    main()
