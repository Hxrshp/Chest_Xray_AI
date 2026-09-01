"""
NIH ChestX-ray14 PyTorch Dataset Implementation
------------------------------------------------
Custom Dataset class loading images and 14-dimensional binary target vectors from CSV manifests.
Converts all images (Grayscale or RGB) into 3-channel RGB tensors for ImageNet backbone compatibility.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from ml.preprocessing.labels import PATHOLOGY_CLASSES, NUM_CLASSES


class NIHChestXrayDataset(Dataset):
    """
    PyTorch Dataset for NIH ChestX-ray14 Multi-Label Pathology Classification.
    """

    def __init__(
        self,
        manifest_path: str,
        transform: Optional[Any] = None,
        raw_images_dir: Optional[str] = None
    ):
        """
        Args:
            manifest_path: Path to train.csv, val.csv, or test.csv manifest.
            transform: PyTorch torchvision transformation pipeline.
            raw_images_dir: Optional override for raw image folder root directory.
        """
        self.manifest_path = Path(manifest_path)

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)
        self.transform = transform
        self.raw_images_dir = Path(raw_images_dir) if raw_images_dir else None

        # Verify required manifest columns
        required_cols = ["image_index", "patient_id"] + PATHOLOGY_CLASSES
        missing_cols = [c for c in required_cols if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Manifest {self.manifest_path} missing required columns: {missing_cols}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        """
        Returns:
            image_tensor: Processed [3, H, W] float32 image tensor.
            target_tensor: 14-dimensional float32 binary label vector.
            image_index: Filename string (e.g. '00000001_000.png').
            patient_id: Patient ID string.
        """
        row = self.df.iloc[idx]
        image_index = str(row["image_index"])
        patient_id = str(row["patient_id"])

        # Determine absolute image path
        if "image_path" in row and pd.notna(row["image_path"]):
            image_path = Path(row["image_path"])
        elif self.raw_images_dir:
            image_path = self.raw_images_dir / image_index
        else:
            image_path = Path("data/raw/images") / image_index

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image file missing at path: {image_path} (Sample index: {idx}, Filename: {image_index})"
            )

        # 1. Load image safely and convert to 3-channel RGB (L -> RGB via channel replication)
        try:
            with Image.open(image_path) as img:
                img_rgb = img.convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load or decode image {image_path}: {e}")

        # 2. Apply transforms
        if self.transform:
            image_tensor = self.transform(img_rgb)
        else:
            image_tensor = torch.from_numpy(np.array(img_rgb)).permute(2, 0, 1).float() / 255.0

        # 3. Extract 14 binary pathology targets
        target_values = row[PATHOLOGY_CLASSES].values.astype(np.float32)
        target_tensor = torch.from_numpy(target_values)

        if target_tensor.shape[0] != NUM_CLASSES:
            raise ValueError(
                f"Invalid target dimension {target_tensor.shape[0]} for image {image_index}. Expected {NUM_CLASSES}."
            )

        return image_tensor, target_tensor, image_index, patient_id
