"""
Count images in data/raw/images using multiple filesystem methods
"""

import os
import glob
from pathlib import Path

p = "data/raw/images"

# Method 1: os.listdir
c1 = len(os.listdir(p))

# Method 2: os.scandir
c2 = sum(1 for _ in os.scandir(p))

# Method 3: pathlib glob
c3 = len(list(Path(p).glob("*.png")))

# Method 4: os.walk
c4 = 0
for root, dirs, files in os.walk(p):
    c4 += len([f for f in files if f.endswith(".png")])

print(f"os.listdir count: {c1}")
print(f"os.scandir count: {c2}")
print(f"pathlib glob count: {c3}")
print(f"os.walk count:    {c4}")
