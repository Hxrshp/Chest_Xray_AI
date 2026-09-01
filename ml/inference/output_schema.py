"""
NIH ChestX-ray14 Inference Output Schema & Data Contracts
---------------------------------------------------------
Defines structured Python classes for deterministic single-image and batch inference responses.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional

MEDICAL_DISCLAIMER = (
    "RESEARCH ONLY DISCLAIMER: This system is an experimental multi-label research baseline for "
    "chest radiograph analysis. It is NOT a clinically validated diagnostic device, certified medical software, "
    "or a replacement for a qualified radiologist. Predictions are statistical model outputs and must never be "
    "used for primary patient diagnosis, automated triage, or direct clinical decision-making."
)


@dataclass
class PathologyPrediction:
    pathology: str
    raw_logit: float
    probability: float
    threshold: Optional[float]
    binary_prediction: Optional[bool]


@dataclass
class PredictionResult:
    image_path: str
    predictions: Dict[str, PathologyPrediction]
    highest_probability_class: str
    highest_probability: float
    model_checkpoint: str
    preprocessing_id: str
    device: str
    disclaimer: str = MEDICAL_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "image_path": self.image_path,
            "highest_probability_class": self.highest_probability_class,
            "highest_probability": float(self.highest_probability),
            "model_checkpoint": self.model_checkpoint,
            "preprocessing_id": self.preprocessing_id,
            "device": self.device,
            "disclaimer": self.disclaimer,
            "pathology_predictions": {}
        }
        for k, v in self.predictions.items():
            res["pathology_predictions"][k] = {
                "raw_logit": float(v.raw_logit),
                "probability": float(v.probability),
                "threshold": float(v.threshold) if v.threshold is not None else None,
                "binary_prediction": bool(v.binary_prediction) if v.binary_prediction is not None else None
            }
        return res

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class BatchPredictionResult:
    total_images: int
    successful_count: int
    failed_count: int
    results: List[PredictionResult]
    failures: Dict[str, str]
    model_checkpoint: str
    device: str
    disclaimer: str = MEDICAL_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_images": self.total_images,
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "results": [r.to_dict() for r in self.results],
            "failures": self.failures,
            "model_checkpoint": self.model_checkpoint,
            "device": self.device,
            "disclaimer": self.disclaimer
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
