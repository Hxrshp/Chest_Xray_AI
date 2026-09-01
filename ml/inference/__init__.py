"""
NIH ChestX-ray14 Production Inference & Explainability Package
--------------------------------------------------------------
Provides unified, type-safe predictor and Grad-CAM explainer interfaces for single and batch radiograph analysis.
"""

from ml.inference.output_schema import PredictionResult, BatchPredictionResult
from ml.inference.preprocessing import preprocess_image
from ml.inference.predictor import Predictor
from ml.inference.explainability import GradCAMExplainer

__all__ = [
    "PredictionResult",
    "BatchPredictionResult",
    "preprocess_image",
    "Predictor",
    "GradCAMExplainer",
]
