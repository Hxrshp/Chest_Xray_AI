"""
NIH ChestX-ray14 Production Predictor Engine
---------------------------------------------
Type-safe, thread-safe, and robust model predictor for single and batch radiograph inference.
Loads selected Phase 6 DenseNet-121 baseline checkpoint and validation-derived thresholds.
"""

import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.builder import build_model
from ml.preprocessing.labels import PATHOLOGY_CLASSES, NUM_CLASSES
from ml.inference.preprocessing import preprocess_image, PREPROCESSING_ID
from ml.inference.output_schema import (
    PredictionResult,
    PathologyPrediction,
    BatchPredictionResult,
    MEDICAL_DISCLAIMER
)


class Predictor:
    """
    Production Predictor class encapsulating Phase 6 DenseNet-121 model and validation thresholds.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        threshold_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None
    ):
        self.project_root = PROJECT_ROOT
        
        # 1. Determine Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 2. Paths
        if checkpoint_path is None:
            checkpoint_path = self.project_root / "checkpoints" / "phase6" / "final" / "best.pth"
            if not checkpoint_path.exists():
                checkpoint_path = self.project_root / "checkpoints" / "phase4" / "best.pth"
        self.checkpoint_path = Path(checkpoint_path)

        if threshold_path is None:
            threshold_path = self.project_root / "data" / "processed" / "phase_5_validation_thresholds.json"
        self.threshold_path = Path(threshold_path)

        # 3. Load Validation Thresholds
        self.thresholds: Dict[str, float] = {}
        if self.threshold_path.exists():
            with open(self.threshold_path, "r", encoding="utf-8") as f:
                t_data = json.load(f)
                for k, v in t_data.items():
                    self.thresholds[k] = float(v.get("selected_threshold", 0.50))

        # 4. Reconstruct DenseNet-121 Model & Load Checkpoint
        model_cfg = {
            "model": {
                "architecture": "densenet121",
                "pretrained": False,
                "num_classes": NUM_CLASSES,
                "dropout_rate": 0.0
            }
        }
        self.model = build_model(model_cfg).to(self.device)
        self.model.eval()

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint missing at path: '{self.checkpoint_path}'")

        ckpt_dict = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        if "model_state_dict" in ckpt_dict:
            self.model.load_state_dict(ckpt_dict["model_state_dict"])
        elif "state_dict" in ckpt_dict:
            self.model.load_state_dict(ckpt_dict["state_dict"])
        else:
            self.model.load_state_dict(ckpt_dict)

        self.model.eval()
        self.ckpt_id = str(self.checkpoint_path)
        self.prep_id = PREPROCESSING_ID

    def predict(self, image_input: Union[str, Path, Image.Image]) -> PredictionResult:
        """
        Executes single-image inference returning structured PredictionResult.
        """
        image_str = str(image_input) if not isinstance(image_input, Image.Image) else "<PIL.Image>"
        img_tensor, _ = preprocess_image(image_input)
        img_tensor = img_tensor.to(self.device, non_blocking=True)

        with torch.inference_mode():
            logits = self.model(img_tensor)
            probs = torch.sigmoid(logits)

        logits_np = logits.cpu().numpy()[0]
        probs_np = probs.cpu().numpy()[0]

        predictions: Dict[str, PathologyPrediction] = {}
        highest_p = -1.0
        highest_class = PATHOLOGY_CLASSES[0]

        for i, class_name in enumerate(PATHOLOGY_CLASSES):
            logit = float(logits_np[i])
            prob = float(probs_np[i])
            thresh = self.thresholds.get(class_name, 0.50)
            binary_dec = bool(prob >= thresh)

            if prob > highest_p:
                highest_p = prob
                highest_class = class_name

            predictions[class_name] = PathologyPrediction(
                pathology=class_name,
                raw_logit=logit,
                probability=prob,
                threshold=thresh,
                binary_prediction=binary_dec
            )

        return PredictionResult(
            image_path=image_str,
            predictions=predictions,
            highest_probability_class=highest_class,
            highest_probability=highest_p,
            model_checkpoint=self.ckpt_id,
            preprocessing_id=self.prep_id,
            device=str(self.device),
            disclaimer=MEDICAL_DISCLAIMER
        )

    def predict_batch(
        self,
        image_inputs: List[Union[str, Path, Image.Image]],
        batch_size: int = 32
    ) -> BatchPredictionResult:
        """
        Executes robust batch inference returning BatchPredictionResult.
        Skips corrupted/missing images while recording error messages.
        """
        results: List[PredictionResult] = []
        failures: Dict[str, str] = {}

        for img_input in image_inputs:
            img_key = str(img_input) if not isinstance(img_input, Image.Image) else f"<PIL.Image_{len(results)}>"
            try:
                res = self.predict(img_input)
                results.append(res)
            except Exception as e:
                failures[img_key] = str(e)

        return BatchPredictionResult(
            total_images=len(image_inputs),
            successful_count=len(results),
            failed_count=len(failures),
            results=results,
            failures=failures,
            model_checkpoint=self.ckpt_id,
            device=str(self.device),
            disclaimer=MEDICAL_DISCLAIMER
        )
