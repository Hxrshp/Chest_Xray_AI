"""
Model Factory Builder for Chest X-Ray Classification Architectures
------------------------------------------------------------------
"""

import torch.nn as nn
from typing import Dict, Any
from ml.models.densenet import DenseNet121ChestXray


def build_model(config: Dict[str, Any]) -> nn.Module:
    """
    Constructs model architecture based on config dictionary or YAML config structure.
    """
    model_cfg = config.get("model", config)
    arch = model_cfg.get("architecture", "densenet121").lower()
    num_classes = model_cfg.get("num_classes", 14)
    pretrained = model_cfg.get("pretrained", True)
    weights_id = model_cfg.get("weights_id", "DenseNet121_Weights.DEFAULT")
    dropout_rate = model_cfg.get("dropout_rate", 0.0)

    if arch == "densenet121":
        model = DenseNet121ChestXray(
            num_classes=num_classes,
            pretrained=pretrained,
            weights_id=weights_id,
            dropout_rate=dropout_rate,
        )
    else:
        raise ValueError(f"Unsupported architecture: {arch}. Currently supported: ['densenet121']")

    # Apply backbone freezing if configured
    freezing_cfg = config.get("freezing", {})
    if freezing_cfg.get("freeze_backbone", False):
        if hasattr(model, "set_backbone_freezing"):
            model.set_backbone_freezing(True)

    return model
