"""
NIH ChestX-ray14 Preprocessing & Robust Image Loader
---------------------------------------------------
Handles safe image decoding (Grayscale, RGB, RGBA), resolution resizing, and ImageNet standardization.
"""

import os
from pathlib import Path
from typing import Tuple, Union
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

