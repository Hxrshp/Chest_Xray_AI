"""
Re-download images_005.tar.gz from official NIH Box repository
--------------------------------------------------------------
"""

import os
import urllib.request
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_PATH = PROJECT_ROOT / "data" / "raw" / "archives" / "images_005.tar.gz"
EXPECTED_BYTES = 3935496531
BOX_URL = "https://nihcc.app.box.com/index.php?rm=box_download_shared_file&shared_name=zx7xbtg7oj9ko9ghmgmsm5zd7tnhhscp&file_id=f_219776556743"


def download_images_005():
    print("=== RE-DOWNLOADING TRUNCATED images_005.tar.gz ===")
    t0 = time.time()
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    req = urllib.request.Request(BOX_URL, headers=headers)
    print(f"Connecting to NIH Box link: {BOX_URL}...")

    with urllib.request.urlopen(req) as resp, open(TARGET_PATH, "wb") as f_out:
        total = int(resp.headers.get("Content-Length", EXPECTED_BYTES))
        print(f"Target size: {total / (1024**3):.2f} GB ({total:,} bytes). Downloading...")
        
        downloaded = 0
        last_log = time.time()

        while chunk := resp.read(8 * 1024 * 1024):
            f_out.write(chunk)
            downloaded += len(chunk)
            if time.time() - last_log >= 5.0:
                pct = (downloaded / total) * 100
                print(f"  Downloaded: {downloaded / (1024**3):.2f} / {total / (1024**3):.2f} GB ({pct:.1f}%)")
                last_log = time.time()

    actual_size = os.path.getsize(TARGET_PATH)
    elapsed = time.time() - t0
    print(f"\nDownload finished in {elapsed:.1f} seconds. Size: {actual_size:,} bytes (Expected: {EXPECTED_BYTES:,})")
    return actual_size == EXPECTED_BYTES


if __name__ == "__main__":
    download_images_005()
