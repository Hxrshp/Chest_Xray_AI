"""
Model Checkpointing & Persistence Management
--------------------------------------------
"""

import os
import torch
import shutil
from typing import Dict, Any, Tuple, Optional
from ml.preprocessing.labels import PATHOLOGY_CLASSES


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = "latest.pth",
    is_best: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    Saves PyTorch checkpoint state dictionary and metadata.
    Automatically updates best.pth if is_best is True.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)

    # Ensure metadata fields exist
    state.setdefault("class_names", PATHOLOGY_CLASSES)
    
    torch.save(state, checkpoint_path)
    best_path = None

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best.pth")
        shutil.copyfile(checkpoint_path, best_path)

    return checkpoint_path, best_path


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Loads checkpoint state dictionary into model, optimizer, and scheduler.
    Returns metadata state dictionary.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at '{checkpoint_path}'")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load optimizer state if available
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state if available
    if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
