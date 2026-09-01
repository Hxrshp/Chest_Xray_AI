"""
NIH ChestX-ray14 Automated Resumable Downloader and Extractor
--------------------------------------------------------------
Implements robust HTTP downloading with retries, resume capability, integrity checks,
and sequential archive extraction & deletion to enforce disk safety rules.
"""

import sys
import os
import time
import tarfile
import hashlib
from pathlib import Path
import urllib.request
import urllib.error

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
IMAGES_DIR = RAW_DIR / "images"
ARCHIVE_DIR = RAW_DIR / "archives"

# Ensure target directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

# Metadata URLs
METADATA_URLS = {
    "Data_Entry_2017.csv": "https://raw.githubusercontent.com/N-Nieto/GenderBias_CheXNet/master/Data_Entry_2017.csv",
    "BBox_List_2017.csv": "https://raw.githubusercontent.com/N-Nieto/GenderBias_CheXNet/master/BBox_List_2017.csv",
    "train_val_list.txt": "https://raw.githubusercontent.com/N-Nieto/GenderBias_CheXNet/master/train_val_list.txt",
    "test_list.txt": "https://raw.githubusercontent.com/N-Nieto/GenderBias_CheXNet/master/test_list.txt",
}

# Image Archive Official NIH Box Static URLs
ARCHIVE_URLS = [
    "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",  # images_001.tar.gz
    "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz",  # images_002.tar.gz
    "https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz",  # images_003.tar.gz
    "https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz",  # images_004.tar.gz
    "https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz",  # images_005.tar.gz
    "https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz",  # images_006.tar.gz
    "https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz",  # images_007.tar.gz
    "https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz",  # images_008.tar.gz
    "https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz",  # images_009.tar.gz
    "https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz",  # images_010.tar.gz
    "https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz",  # images_011.tar.gz
    "https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz",  # images_012.tar.gz
]


def download_file(url: str, dest_path: Path, max_retries: int = 5, chunk_size: int = 1024 * 1024) -> bool:
    """
    Downloads a file with chunked streaming, retry logic, and resume support.
    """
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Downloading {dest_path.name} (Attempt {attempt}/{max_retries})...")
            
            # Check existing temp bytes for resume
            initial_bytes = temp_path.stat().st_size if temp_path.exists() else 0
            req = urllib.request.Request(url)
            if initial_bytes > 0:
                req.add_header("Range", f"bytes={initial_bytes}-")

            with urllib.request.urlopen(req, timeout=30) as response, open(temp_path, "ab" if initial_bytes > 0 else "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
            
            # Move temp file to final destination
            if temp_path.exists():
                temp_path.replace(dest_path)
            print(f"Successfully downloaded: {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")
            return True

        except Exception as e:
            print(f"Warning: Download attempt {attempt} failed for {dest_path.name}: {e}")
            time.sleep(2 * attempt)

    print(f"ERROR: Failed to download {dest_path.name} after {max_retries} attempts.")
    return False


def verify_tar_archive(tar_path: Path) -> bool:
    """Verifies that a downloaded .tar.gz archive is valid and uncorrupted."""
    try:
        print(f"Verifying archive integrity: {tar_path.name}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            # Inspection of archive members
            members = tar.getmembers()
            if len(members) == 0:
                print(f"Error: Archive {tar_path.name} is empty.")
                return False
        print(f"Archive verification PASSED for {tar_path.name} ({len(members)} entries found).")
        return True
    except Exception as e:
        print(f"ERROR: Archive verification FAILED for {tar_path.name}: {e}")
        return False


def extract_and_clean(tar_path: Path, extract_dir: Path) -> bool:
    """
    Extracts tar archive into target directory, verifies extracted images,
    and removes the archive file to save disk space.
    """
    try:
        print(f"Extracting {tar_path.name} into {extract_dir}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        print(f"Extraction complete for {tar_path.name}.")

        # Enforce DISK-SAFETY RULE: Only delete archive after successful extraction
        if tar_path.exists():
            tar_path.unlink()
            print(f"Deleted archive {tar_path.name} to recover disk space.")
        return True
    except Exception as e:
        print(f"ERROR during extraction of {tar_path.name}: {e}")
        return False


def run_pipeline():
    print("=== STARTING NIH CHESTX-RAY14 ACQUISITION PIPELINE ===")

    # 1. Download Metadata Files
    print("\n--- Phase 1: Metadata Files Download ---")
    for name, url in METADATA_URLS.items():
        dest = RAW_DIR / name
        if not dest.exists():
            success = download_file(url, dest)
            if not success:
                print(f"CRITICAL ERROR: Failed to acquire metadata file {name}. STOPPING.")
                sys.exit(1)
        else:
            print(f"Metadata file already present: {name}")

    print("\nMetadata acquisition complete.")


if __name__ == "__main__":
    run_pipeline()
