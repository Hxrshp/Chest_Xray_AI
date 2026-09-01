"""
NIH ChestX-ray14 Explanation Service Module
-------------------------------------------
Service wrapper for Grad-CAM visual heatmap generation.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Union, Optional
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.explainability import GradCAMExplainer
from app.services.inference_service import get_predictor


def generate_gradcam_explanation(
    image_input: Union[str, Path, Image.Image],
    target_class: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Generates class activation heatmap and colorized overlay for specified pathology.
    """
    predictor = get_predictor()
    explainer = GradCAMExplainer(predictor)
    return explainer.explain(image_input, target_class=target_class, output_dir=output_dir)
