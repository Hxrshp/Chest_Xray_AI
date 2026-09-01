"""
NIH ChestX-ray14 Preprocessing Diagnostic Visualization Utility
----------------------------------------------------------------
Generates a small diagnostic sample of 4-8 preprocessed images saved under
docs/phase_2d_visualizations/ to visually inspect anatomy preservation, resizing,
grayscale to RGB conversion, normalization, and training augmentation behavior.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

TRAIN_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifests" / "train.csv"
VIS_DIR = PROJECT_ROOT / "docs" / "phase_2d_visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)

from ml.preprocessing.dataset import NIHChestXrayDataset
from ml.preprocessing.transforms import get_transforms, IMAGENET_MEAN, IMAGENET_STD


def denormalize_tensor(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Denormalizes a [3, H, W] tensor back to [0, 1] RGB numpy array for display."""
    t = tensor.clone()
    for c in range(3):
        t[c] = t[c] * std[c] + mean[c]
    t = torch.clamp(t, 0.0, 1.0)
    return t.permute(1, 2, 0).numpy()


def generate_visualizations():
    print("=== GENERATING PREPROCESSING DIAGNOSTIC VISUALIZATIONS ===")
    if not TRAIN_MANIFEST.exists():
        print(f"ERROR: Train manifest missing: {TRAIN_MANIFEST}")
        sys.exit(1)

    eval_transform = get_transforms(image_size=(320, 320), is_training=False)
    aug_transform = get_transforms(image_size=(320, 320), is_training=True, enable_augmentation=True)

    dataset_eval = NIHChestXrayDataset(manifest_path=str(TRAIN_MANIFEST), transform=eval_transform)
    dataset_aug = NIHChestXrayDataset(manifest_path=str(TRAIN_MANIFEST), transform=aug_transform)

    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    fig.suptitle("NIH ChestX-ray14 Phase 2D Preprocessing Diagnostics", fontsize=14, fontweight="bold")

    indices = [0, 10, 25, 50]
    for row_idx, sample_idx in enumerate(indices):
        img_tensor_eval, target, img_idx, patient_id = dataset_eval[sample_idx]
        img_tensor_aug, _, _, _ = dataset_aug[sample_idx]

        raw_path = Path("data/raw/images") / img_idx
        raw_pil = Image.open(raw_path)

        # 1. Raw Image
        axes[row_idx, 0].imshow(raw_pil, cmap="gray" if raw_pil.mode == "L" else None)
        axes[row_idx, 0].set_title(f"Raw: {img_idx}\n({raw_pil.width}x{raw_pil.height}, {raw_pil.mode})", fontsize=10)
        axes[row_idx, 0].axis("off")

        # 2. Eval Preprocessed (320x320 RGB Normalized)
        img_eval_np = denormalize_tensor(img_tensor_eval)
        axes[row_idx, 1].imshow(img_eval_np)
        axes[row_idx, 1].set_title(f"Eval Pipeline\n(3x320x320 RGB)", fontsize=10)
        axes[row_idx, 1].axis("off")

        # 3. Augmented Training Preprocessed
        img_aug_np = denormalize_tensor(img_tensor_aug)
        axes[row_idx, 2].imshow(img_aug_np)
        axes[row_idx, 2].set_title(f"Augmented Train\n(Mild Affine/Jitter)", fontsize=10)
        axes[row_idx, 2].axis("off")

    plt.tight_layout()
    output_png = VIS_DIR / "preprocessing_sample_grid.png"
    plt.savefig(output_png, dpi=150)
    plt.close()

    print(f"Saved diagnostic grid plot to {output_png}")
    return output_png


if __name__ == "__main__":
    generate_visualizations()
