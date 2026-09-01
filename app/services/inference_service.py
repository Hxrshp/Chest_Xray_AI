"""
NIH ChestX-ray14 Inference Service Module
------------------------------------------
Provides thread-safe, cached loading of the Phase 7 Predictor engine.
"""

import sys
from pathlib import Path
from typing import Union
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.predictor import Predictor
from ml.inference.output_schema import PredictionResult
from app.config import CHECKPOINT_PATH, THRESHOLD_PATH

# Global Predictor Instance for non-Streamlit environments
_predictor_instance = None


def get_predictor() -> Predictor:
    """
    Returns cached instance of Predictor. Reuses existing model instance in memory.
    """
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = Predictor(
            checkpoint_path=CHECKPOINT_PATH,
            threshold_path=THRESHOLD_PATH
        )
    return _predictor_instance


def run_inference(image_input: Union[str, Path, Image.Image]) -> PredictionResult:
    """
    Executes single-image radiograph classification returning PredictionResult.
    """
    predictor = get_predictor()
    return predictor.predict(image_input)
