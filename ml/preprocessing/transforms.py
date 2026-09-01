"""
NIH ChestX-ray14 Medically Conservative torchvision Transforms
--------------------------------------------------------------
Implements ImageNet-compatible transforms for transfer learning CNN backbones.
- Train: Mild affine (rotation <= 7 deg, scale 0.95-1.05), mild brightness/contrast.
- Val/Test: Deterministic Resize -> ToTensor -> Normalize (zero randomness).
- Horizontal flipping disabled by default to preserve anatomical integrity (dextrocardia).
"""

from typing import Tuple, List, Optional
import torchvision.transforms as T
import torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(
    image_size: Tuple[int, int] = (320, 320),
    is_training: bool = False,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
    enable_augmentation: bool = True,
    rotation_degrees: float = 7.0,
    translation: Tuple[float, float] = (0.05, 0.05),
    scale_range: Tuple[float, float] = (0.95, 1.05),
    brightness: float = 0.1,
    contrast: float = 0.1,
    horizontal_flip: bool = False
) -> T.Compose:
    """
    Constructs PyTorch torchvision transformation pipeline.

    Args:
        image_size: Target (height, width) tuple.
        is_training: If True, applies mild medical augmentation. If False, deterministic.
        mean: ImageNet normalization mean per channel.
        std: ImageNet normalization standard deviation per channel.
        enable_augmentation: Enables augmentation during training.
        rotation_degrees: Maximum small rotation in degrees.
        translation: Small random translation fraction.
        scale_range: Small random scaling factor range.
        brightness: Mild brightness jitter factor.
        contrast: Mild contrast jitter factor.
        horizontal_flip: If True, enables random horizontal flipping (disabled by default).

    Returns:
        T.Compose: Configured torchvision transform pipeline.
    """
    transform_list = []

    # 1. Resize to target square resolution
    transform_list.append(T.Resize(image_size, antialias=True))

    # 2. Apply Training-Only Medically Conservative Augmentations
    if is_training and enable_augmentation:
        if rotation_degrees > 0 or translation != (0.0, 0.0) or scale_range != (1.0, 1.0):
            transform_list.append(
                T.RandomAffine(
                    degrees=rotation_degrees,
                    translate=translation,
                    scale=scale_range,
                    fill=0
                )
            )
        if brightness > 0 or contrast > 0:
            transform_list.append(
                T.ColorJitter(
                    brightness=brightness,
                    contrast=contrast
                )
            )
        if horizontal_flip:
            transform_list.append(T.RandomHorizontalFlip(p=0.5))

    # 3. Convert PIL Image (0-255) to FloatTensor (0.0-1.0)
    transform_list.append(T.ToTensor())

    # 4. Normalize using ImageNet statistics
    transform_list.append(T.Normalize(mean=mean, std=std))

    return T.Compose(transform_list)
