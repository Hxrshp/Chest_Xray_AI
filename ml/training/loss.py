"""
Multi-Label Loss Function Builder & Class Imbalance Weighting
-------------------------------------------------------------
"""

import json
import os
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from ml.preprocessing.labels import PATHOLOGY_CLASSES


def load_train_pos_weights(
    class_stats_path: str = "data/processed/class_statistics.json",
    device: torch.device = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Loads train-only positive weights (bce_pos_weights) from class_statistics.json.
    Ensures exact 14-class alignment with PATHOLOGY_CLASSES.
    """
    if not os.path.exists(class_stats_path):
        raise FileNotFoundError(
            f"Class statistics file not found at '{class_stats_path}'. "
            "Please run scripts/compute_class_statistics.py first."
        )

    with open(class_stats_path, "r") as f:
        data = json.load(f)

    bce_weights_dict = data.get("bce_pos_weights", {})
    weights_list = []

    for cls in PATHOLOGY_CLASSES:
        if cls not in bce_weights_dict:
            raise KeyError(f"Class '{cls}' missing from bce_pos_weights in {class_stats_path}")
        w = float(bce_weights_dict[cls])
        if torch.isnan(torch.tensor(w)) or torch.isinf(torch.tensor(w)) or w <= 0.0:
            raise ValueError(f"Invalid positive weight for class '{cls}': {w}")
        weights_list.append(w)

    pos_weight_tensor = torch.tensor(weights_list, dtype=torch.float32)
    if device is not None:
        pos_weight_tensor = pos_weight_tensor.to(device)

    return pos_weight_tensor, bce_weights_dict


def get_loss_function(
    config: Dict[str, Any],
    class_stats_path: str = "data/processed/class_statistics.json",
    device: torch.device = None,
) -> Tuple[nn.Module, torch.Tensor]:
    """
    Constructs the multi-label loss criterion based on config parameters.
    Default: nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor).
    """
    loss_cfg = config.get("loss", {})
    loss_type = loss_cfg.get("type", "weighted_bce").lower()
    use_pos_weight = loss_cfg.get("use_pos_weight", True)

    if loss_type == "weighted_bce" and use_pos_weight:
        pos_weight_tensor, _ = load_train_pos_weights(class_stats_path, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    elif loss_type == "bce" or not use_pos_weight:
        pos_weight_tensor = torch.ones(len(PATHOLOGY_CLASSES), dtype=torch.float32)
        if device is not None:
            pos_weight_tensor = pos_weight_tensor.to(device)
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported loss type: '{loss_type}'")

    return criterion, pos_weight_tensor
