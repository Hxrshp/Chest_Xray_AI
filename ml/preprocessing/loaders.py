"""
NIH ChestX-ray14 DataLoader Builder Utilities
----------------------------------------------
Constructs PyTorch DataLoaders for train, validation, and test splits.
Applies shuffle=True for train, and deterministic shuffle=False for val and test.
"""

import os
from typing import Dict, Tuple, Optional
import torch
from torch.utils.data import DataLoader

from ml.preprocessing.dataset import NIHChestXrayDataset
from ml.preprocessing.transforms import get_transforms


def create_dataloaders(
    train_manifest: str = "data/processed/manifests/train.csv",
    val_manifest: str = "data/processed/manifests/val.csv",
    test_manifest: str = "data/processed/manifests/test.csv",
    image_size: Tuple[int, int] = (320, 320),
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    enable_augmentation: bool = True,
    rotation_degrees: float = 7.0,
    translation: Tuple[float, float] = (0.05, 0.05),
    scale_range: Tuple[float, float] = (0.95, 1.05),
    brightness: float = 0.1,
    contrast: float = 0.1,
    horizontal_flip: bool = False,
    seed: int = 42
) -> Dict[str, DataLoader]:
    """
    Creates PyTorch DataLoaders for train, val, and test splits.

    Returns:
        Dict[str, DataLoader]: Dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    # 1. Transforms
    train_transform = get_transforms(
        image_size=image_size,
        is_training=True,
        enable_augmentation=enable_augmentation,
        rotation_degrees=rotation_degrees,
        translation=translation,
        scale_range=scale_range,
        brightness=brightness,
        contrast=contrast,
        horizontal_flip=horizontal_flip
    )

    eval_transform = get_transforms(
        image_size=image_size,
        is_training=False
    )

    # 2. Datasets
    train_dataset = NIHChestXrayDataset(manifest_path=train_manifest, transform=train_transform)
    val_dataset = NIHChestXrayDataset(manifest_path=val_manifest, transform=eval_transform)
    test_dataset = NIHChestXrayDataset(manifest_path=test_manifest, transform=eval_transform)

    # Set generator for train loader shuffling reproducibility
    g = torch.Generator()
    g.manual_seed(seed)

    # 3. DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }


def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience wrapper to build train, val, and test dataloaders from config dictionary.
    """
    dataset_cfg = config.get("dataset", {})
    prep_cfg = config.get("preprocessing", {})
    aug_cfg = config.get("augmentation", {})
    loader_cfg = config.get("dataloader", {})
    repro_cfg = config.get("reproducibility", {})

    loaders_dict = create_dataloaders(
        train_manifest=dataset_cfg.get("train_manifest", "data/processed/manifests/train.csv"),
        val_manifest=dataset_cfg.get("val_manifest", "data/processed/manifests/val.csv"),
        test_manifest=dataset_cfg.get("test_manifest", "data/processed/manifests/test.csv"),
        image_size=tuple(prep_cfg.get("image_size", [320, 320])),
        batch_size=int(loader_cfg.get("batch_size", 32)),
        num_workers=int(loader_cfg.get("num_workers", 0)),
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        enable_augmentation=bool(aug_cfg.get("enable_training_augmentation", True)),
        rotation_degrees=float(aug_cfg.get("rotation_degrees", 7.0)),
        translation=tuple(aug_cfg.get("translation", [0.05, 0.05])),
        scale_range=tuple(aug_cfg.get("scale_range", [0.95, 1.05])),
        brightness=float(aug_cfg.get("brightness", 0.1)),
        contrast=float(aug_cfg.get("contrast", 0.1)),
        horizontal_flip=bool(aug_cfg.get("horizontal_flip", False)),
        seed=int(repro_cfg.get("seed", 42)),
    )
    return loaders_dict["train"], loaders_dict["val"], loaders_dict["test"]

