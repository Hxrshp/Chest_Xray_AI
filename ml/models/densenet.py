"""
DenseNet-121 Architecture for Multi-Label Chest X-Ray Classification
----------------------------------------------------------------------
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import DenseNet121_Weights


class DenseNet121ChestXray(nn.Module):
    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        weights_id: str = "DenseNet121_Weights.DEFAULT",
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.weights_id = weights_id if pretrained else None

        if pretrained:
            if weights_id == "DenseNet121_Weights.DEFAULT":
                weights = DenseNet121_Weights.DEFAULT
            else:
                weights = DenseNet121_Weights.DEFAULT
            self.backbone = models.densenet121(weights=weights)
        else:
            self.backbone = models.densenet121(weights=None)

        # Retrieve feature dimension of classifier
        num_features = self.backbone.classifier.in_features

        # Replace classifier with custom 14-class linear head (producing raw logits)
        if dropout_rate > 0.0:
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_features, num_classes),
            )
        else:
            self.backbone.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw, unscaled logits [B, num_classes].
        Sigmoid is NOT applied inside forward to preserve numerical stability for BCEWithLogitsLoss.
        """
        return self.backbone(x)

    def set_backbone_freezing(self, freeze: bool = True) -> None:
        """
        Freezes or unfreezes backbone parameters (features block).
        """
        for param in self.backbone.features.parameters():
            param.requires_grad = not freeze

    def get_parameter_counts(self) -> dict:
        """
        Returns total, trainable, and frozen parameter counts.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        return {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": frozen_params,
        }
