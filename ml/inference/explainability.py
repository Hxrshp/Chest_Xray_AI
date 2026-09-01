"""
NIH ChestX-ray14 Class Activation Mapping (CAM) Visual Explainability Module
----------------------------------------------------------------------------
Generates mathematically exact Class Activation Maps (CAM) for DenseNet-121 multi-label model.
Uses classifier linear weights W_{c,k} and final feature maps from model.backbone.features.
Applies thoracic lung field spatial masking to eliminate border edge-padding artifacts.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing.labels import PATHOLOGY_CLASSES
from ml.inference.preprocessing import preprocess_image
from ml.inference.predictor import Predictor


class GradCAMExplainer:
    """
    Class Activation Mapping (CAM) Visual Explainer for DenseNet-121 architecture.
    Computes exact weighted feature map activations using linear classifier weights.
    """

    def __init__(self, predictor: Predictor, target_layer: Optional[nn.Module] = None):
        self.predictor = predictor
        self.model = predictor.model
        self.device = predictor.device
        self.classifier_weights = self.model.backbone.classifier.weight.detach().cpu()  # [14, 1024]

    def explain(
        self,
        image_input: Union[str, Path, Image.Image],
        target_class: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        noise_threshold: float = 0.25
    ) -> Dict[str, Any]:
        """
        Generates Class Activation Map (CAM) heatmap and colorized overlay for the target pathology.
        """
        # 1. Preprocess Image
        img_tensor, orig_pil = preprocess_image(image_input)
        img_tensor = img_tensor.to(self.device)
        orig_w, orig_h = orig_pil.size

        # 2. Forward Pass for Features & Logits
        self.model.eval()
        with torch.no_grad():
            features = self.model.backbone.features(img_tensor)  # [1, 1024, H_feat, W_feat]
            features_relu = F.relu(features[0])  # [1024, H_feat, W_feat]
            
            # Predict Logits & Probabilities
            out_pooled = F.adaptive_avg_pool2d(features, (1, 1))
            out_flat = torch.flatten(out_pooled, 1)
            logits = self.model.backbone.classifier(out_flat)[0]
            probs = torch.sigmoid(logits)

        # 3. Determine Target Pathology Index
        if target_class is None:
            c_idx = int(torch.argmax(probs).item())
            target_class = PATHOLOGY_CLASSES[c_idx]
        else:
            if target_class not in PATHOLOGY_CLASSES:
                raise ValueError(f"Invalid pathology class: '{target_class}'. Must be one of {PATHOLOGY_CLASSES}")
            c_idx = PATHOLOGY_CLASSES.index(target_class)

        target_prob = float(probs[c_idx].item())

        # 4. Compute Positive-Weighted CAM for Target Class
        class_weights = F.relu(self.classifier_weights[c_idx]).to(self.device).view(-1, 1, 1)  # [1024, 1, 1]
        cam_tensor = torch.sum(class_weights * features_relu, dim=0)  # [H_feat, W_feat]

        cam_np = cam_tensor.cpu().numpy()

        # 5. Normalize CAM to [0, 1]
        if cam_np.max() > cam_np.min():
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        else:
            cam_np = np.zeros_like(cam_np)

        # 6. Resize Heatmap to Original Image Dimensions
        cam_pil = Image.fromarray(np.uint8(255 * cam_np)).resize((orig_w, orig_h), resample=Image.BILINEAR)
        heatmap_norm = np.array(cam_pil, dtype=np.float32) / 255.0

        # 7. Thoracic Lung Field Masking: Zero out outer background air & image margins
        orig_gray = np.array(orig_pil.convert("L"), dtype=np.float32) / 255.0
        
        # Mask out dark outer background air (pixels < 10% intensity)
        body_mask = (orig_gray > 0.10).astype(np.float32)
        
        # Mask out outer 3% border margins of the image canvas
        margin_mask = np.zeros_like(orig_gray, dtype=np.float32)
        h_m = max(1, int(orig_h * 0.03))
        w_m = max(1, int(orig_w * 0.03))
        margin_mask[h_m:orig_h-h_m, w_m:orig_w-w_m] = 1.0

        heatmap_norm = heatmap_norm * body_mask * margin_mask
        if heatmap_norm.max() > 0:
            heatmap_norm = (heatmap_norm - heatmap_norm.min()) / (heatmap_norm.max() - heatmap_norm.min() + 1e-8)

        # 8. Create Colorized Jet Overlay
        cmap = plt.get_cmap("jet")
        colored_heatmap = cmap(heatmap_norm)[:, :, :3]  # [H, W, 3]
        colored_heatmap_pil = Image.fromarray(np.uint8(colored_heatmap * 255))

        # Alpha mask: zero activation = natural grayscale X-ray, high activation = jet overlay
        alpha_mask = (heatmap_norm ** 1.2)[:, :, None]
        orig_rgb = np.array(orig_pil.convert("RGB"), dtype=np.float32) / 255.0
        
        blended_np = orig_rgb * (1.0 - 0.45 * alpha_mask) + colored_heatmap * (0.45 * alpha_mask)
        blended_np = np.clip(blended_np * 255.0, 0, 255).astype(np.uint8)
        overlay_pil = Image.fromarray(blended_np)

        # 9. Optionally Save Visualization Files
        saved_paths = {}
        if output_dir is not None:
            out_path = Path(output_dir)
            os.makedirs(out_path, exist_ok=True)
            orig_file = out_path / f"original_{target_class}.png"
            heat_file = out_path / f"heatmap_{target_class}.png"
            over_file = out_path / f"overlay_{target_class}.png"

            orig_pil.save(orig_file)
            colored_heatmap_pil.save(heat_file)
            overlay_pil.save(over_file)

            saved_paths = {
                "original_path": str(orig_file),
                "heatmap_path": str(heat_file),
                "overlay_path": str(over_file)
            }

        return {
            "target_class": target_class,
            "target_probability": target_prob,
            "heatmap": heatmap_norm,
            "heatmap_pil": colored_heatmap_pil,
            "overlay_pil": overlay_pil,
            "saved_paths": saved_paths,
            "disclaimer": "ATTENTION VISUALIZATION DISCLAIMER: Heatmap represents model feature activation, NOT a proven pathological diagnosis."
        }
