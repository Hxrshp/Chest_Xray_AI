"""
NIH ChestX-ray14 Preprocessing & Robust Image Loader
---------------------------------------------------
Handles safe image decoding (Grayscale, RGB, RGBA), resolution resizing, and ImageNet standardization.
"""

import os
from pathlib import Path
from typing import Tuple, Union, Optional
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T


PREPROCESSING_ID = "Medical_ChestXRay_Scale224_Norm_m1024_p1024"


def load_and_validate_image(image_input: Union[str, Path, Image.Image]) -> Image.Image:
    """
    Validates file existence, non-zero byte size, and decodes Grayscale/RGB/RGBA into 3-channel RGB PIL Image.
    """
    if isinstance(image_input, Image.Image):
        img_rgb = image_input.convert("RGB")
        return img_rgb

    file_path = Path(image_input)
    if not file_path.exists():
        raise FileNotFoundError(f"Image file missing at path: '{file_path}'")

    if file_path.stat().st_size == 0:
        raise ValueError(f"Image file '{file_path}' is empty (0 bytes)")

    try:
        with Image.open(file_path) as img:
            img.verify()
        
        # Re-open after verify()
        with Image.open(file_path) as img:
            img_rgb = img.convert("RGB")
            # Force dimension evaluation to ensure image data is valid
            _ = img_rgb.size
            return img_rgb
    except Exception as e:
        raise ValueError(f"Failed to load or decode image '{file_path}': {e}")


def preprocess_image(
    image_input: Union[str, Path, Image.Image],
    image_size: Tuple[int, int] = (224, 224)
) -> Tuple[torch.Tensor, Image.Image]:
    """
    Loads, validates, and transforms image into [1, 3, H, W] float32 tensor scaled to [-1024, 1024]
    along with original PIL image.
    """
    pil_img = load_and_validate_image(image_input)
    # Resize to target resolution
    img_resized = pil_img.resize(image_size, resample=Image.BILINEAR)
    img_np = np.array(img_resized, dtype=np.float32)
    
    # Clinical medical normalization: scale [0, 255] to [-1024, 1024]
    img_norm = (2.0 * (img_np / 255.0) - 1.0) * 1024.0
    
    # [H, W, 3] -> [1, 3, H, W]
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()
    return img_tensor, pil_img


def validate_radiograph_modality(pil_img: Image.Image) -> Tuple[bool, Optional[str]]:
    """
    Validates that the input image is a legitimate monochrome chest radiograph.
    Strictly blocks Out-of-Distribution (OOD) inputs like color photographs,
    scenery, vehicles, or corrupted images.

    Returns:
        (is_valid: bool, error_reason: Optional[str])
    """
    img_rgb = pil_img.convert("RGB")
    arr = np.array(img_rgb, dtype=np.float32)

    # Check 1: Minimum resolution
    w, h = pil_img.size
    if w < 100 or h < 100:
        return False, f"Image resolution is too low ({w}×{h}px). Minimum required is 100×100px."

    # Check 2: Contrast / Standard Deviation (blank or solid images)
    gray = np.mean(arr, axis=2)
    std_dev = float(np.std(gray))
    if std_dev < 10.0:
        return False, f"Image has insufficient contrast (std: {std_dev:.1f}). Image appears blank or corrupted."

    # Check 3: Color Saturation / Inter-Channel Discrepancy (Natural photos vs Monochrome X-rays)
    # Real X-rays have near-identical R, G, B channels (channel difference near 0)
    channel_diff = float(np.mean(
        np.abs(arr[:, :, 0] - arr[:, :, 1]) +
        np.abs(arr[:, :, 1] - arr[:, :, 2]) +
        np.abs(arr[:, :, 0] - arr[:, :, 2])
    ) / 3.0)

    max_c = np.max(arr, axis=2)
    min_c = np.min(arr, axis=2)
    # Average saturation for non-black pixels
    mask = max_c > 15
    if np.any(mask):
        sat_pct = float(np.mean((max_c[mask] - min_c[mask]) / (max_c[mask] + 1e-5)) * 100.0)
    else:
        sat_pct = 0.0

    # High color saturation means it's a color photo, not an X-ray
    if channel_diff > 12.0 or sat_pct > 15.0:
        return False, "Color photograph detected. This system only analyzes monochrome chest X-rays."

    return True, None

