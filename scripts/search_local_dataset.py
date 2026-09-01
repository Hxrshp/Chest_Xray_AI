"""
Local Dataset Search Script
---------------------------
Searches reasonable local machine directories for NIH ChestX-ray14 dataset files
without modifying or deleting anything.
"""

import os
from pathlib import Path

SEARCH_TARGETS = [
    "Data_Entry_2017.csv",
    "BBox_List_2017.csv",
    "images_01.tar.gz",
    "images_001.tar.gz",
    "ChestXray-NIHCC",
    "chest_xray",
    "nih-chest-xrays"
]

CANDIDATE_DIRS = [
    Path(r"D:\XRAY-ABSTRACT"),
    Path(r"D:\data"),
    Path(r"D:\datasets"),
    Path(r"C:\Users\Harsha\Downloads"),
    Path(r"C:\Users\Harsha\Desktop"),
    Path(r"C:\Users\Harsha\Documents"),
    Path(r"C:\data"),
]


def search_local():
    found_locations = []

    for base in CANDIDATE_DIRS:
        if not base.exists():
            continue
        
        print(f"Scanning directory: {base}")
        try:
            for root, dirs, files in os.walk(base):
                # Don't recurse into .venv, .git, node_modules
                dirs[:] = [d for d in dirs if d not in ('.venv', '.git', 'node_modules', '$RECYCLE.BIN')]
                
                for f in files:
                    if f in SEARCH_TARGETS or any(target.lower() in f.lower() for target in ["data_entry_2017", "chestxray"]):
                        match_path = Path(root) / f
                        found_locations.append(str(match_path))
        except Exception as e:
            print(f"Error scanning {base}: {e}")

    print("\n=== SEARCH RESULTS ===")
    if found_locations:
        print(f"Dataset files FOUND at {len(found_locations)} location(s):")
        for loc in found_locations:
            print(f" - {loc}")
    else:
        print("NO NIH ChestX-ray14 dataset files found in searched candidate directories.")


if __name__ == "__main__":
    search_local()
