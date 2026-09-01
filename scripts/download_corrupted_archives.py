"""
Re-download truncated archives (images_006.tar.gz and images_009.tar.gz) from official NIH Box repository
---------------------------------------------------------------------------------------------------------
"""

import os
import urllib.request
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVES_DIR = PROJECT_ROOT / "data" / "raw" / "archives"

FILES_TO_DOWNLOAD = [
    {
        "name": "images_006.tar.gz",
        "file_id": "219777758783",
        "expected_bytes": 3986301172
    },
    {
        "name": "images_009.tar.gz",
        "file_id": "219782291318",
        "expected_bytes": 4111327929
    }
]


def download_corrupted():
    print("=== RE-DOWNLOADING TRUNCATED ARCHIVES FROM NIH BOX ===")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for item in FILES_TO_DOWNLOAD:
        fname = item["name"]
        fid = item["file_id"]
        exp = item["expected_bytes"]
        target = ARCHIVES_DIR / fname

        if target.exists() and os.path.getsize(target) == exp:
            print(f"Skipping {fname}: already matches expected size ({exp:,} bytes).")
            continue

        url = f"https://nihcc.app.box.com/index.php?rm=box_download_shared_file&shared_name=zx7xbtg7oj9ko9ghmgmsm5zd7tnhhscp&file_id=f_{fid}"
        print(f"\nDownloading {fname} ({exp / (1024**3):.2f} GB) from {url}...")
        t0 = time.time()

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp, open(target, "wb") as f_out:
            total = int(resp.headers.get("Content-Length", exp))
            downloaded = 0
            last_log = time.time()

            while chunk := resp.read(8 * 1024 * 1024):
                f_out.write(chunk)
                downloaded += len(chunk)
                if time.time() - last_log >= 5.0:
                    pct = (downloaded / total) * 100
                    print(f"  [{fname}] {downloaded / (1024**3):.2f} / {total / (1024**3):.2f} GB ({pct:.1f}%)")
                    last_log = time.time()

        act = os.path.getsize(target)
        print(f"Finished {fname} in {time.time() - t0:.1f}s. Size: {act:,} bytes (Expected: {exp:,})")

    print("\nAll corrupted archives re-downloaded successfully!")


if __name__ == "__main__":
    download_corrupted()
