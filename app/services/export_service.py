"""
NIH ChestX-ray14 Result Export Service
--------------------------------------
Generates machine-readable JSON export payloads containing predictions, metadata, SHA-256 hashes, and safety disclaimers.
"""

import sys
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.output_schema import PredictionResult
from app.config import MODEL_METRICS, MEDICAL_DISCLAIMER_FULL


def create_export_payload(
    result: PredictionResult,
    image_bytes: Optional[bytes] = None,
    inference_time_sec: Optional[float] = None
) -> Dict[str, Any]:
    """
    Creates structured JSON export payload.
    """
    img_hash = hashlib.sha256(image_bytes).hexdigest() if image_bytes else "N/A"

    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "image_metadata": {
            "image_identifier": Path(result.image_path).name,
            "sha256_hash": img_hash,
        },
        "model_metadata": {
            "architecture": MODEL_METRICS["architecture"],
            "selected_experiment": MODEL_METRICS["selected_experiment"],
            "checkpoint_path": result.model_checkpoint,
            "preprocessing_identifier": result.preprocessing_id,
            "inference_device": result.device,
            "inference_time_sec": round(inference_time_sec, 4) if inference_time_sec is not None else None,
            "benchmark_val_macro_auroc": MODEL_METRICS["val_macro_auroc"],
            "benchmark_test_macro_auroc": MODEL_METRICS["test_macro_auroc"],
        },
        "predictions": {
            "highest_probability_class": result.highest_probability_class,
            "highest_probability": float(result.highest_probability),
            "pathologies": {}
        },
        "disclaimer": MEDICAL_DISCLAIMER_FULL
    }

    for c_name, pred in result.predictions.items():
        payload["predictions"]["pathologies"][c_name] = {
            "raw_logit": round(float(pred.raw_logit), 4),
            "probability": round(float(pred.probability), 4),
            "validation_threshold": round(float(pred.threshold), 4) if pred.threshold is not None else None,
            "model_prediction": "Positive" if pred.binary_prediction else "Negative"
        }

    return payload
