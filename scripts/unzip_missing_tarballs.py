"""
Extract missing tarballs from images-selected.zip into data/raw/archives/
-------------------------------------------------------------------------
"""

import os
import zipfile
import shutil
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVES_DIR = PROJECT_ROOT / "data" / "raw" / "archives"
ZIP_PATH = ARCHIVES_DIR / "images-selected.zip"


def unzip_missing():
    print("=== UNZIPPING MISSING TARBALLS FROM images-selected.zip ===")
    t0 = time.time()
    
    if not ZIP_PATH.exists():
        print(f"ERROR: {ZIP_PATH} does not exist.")
        return

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        members = [m for m in z.namelist() if m.endswith(".tar.gz")]
        print(f"Found {len(members)} tarballs inside {ZIP_PATH.name}.")
        for m in members:
            fname = Path(m).name
            dest_file = ARCHIVES_DIR / fname
            if not dest_file.exists():
                print(f"Unzipping {fname}...")
                z.extract(m, ARCHIVES_DIR)
                extracted_file = ARCHIVES_DIR / m
                if extracted_file.exists() and extracted_file.resolve() != dest_file.resolve():
                    shutil.move(str(extracted_file), str(dest_file))
                    # Clean up empty parent directories if created
                    if extracted_file.parent != ARCHIVES_DIR:
                        try:
                            extracted_file.parent.rmdir()
                        except Exception:
                            pass

    current_tarballs = sorted(list(ARCHIVES_DIR.glob("images_*.tar.gz")))
    print(f"\nUnzipping complete in {time.time() - t0:.2f} seconds.")
    print(f"Total tarballs in {ARCHIVES_DIR}: {len(current_tarballs)} / 12.")


if __name__ == "__main__":
    unzip_missing()
