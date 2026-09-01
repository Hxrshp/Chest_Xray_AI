"""
Phase 7D — Grad-CAM Explanation CLI Tool
---------------------------------------
Generates class activation heatmaps and colorized overlays for visual feature explainability.

Usage:
    python scripts/generate_explanation.py --image data/raw/images/00000001_000.png [--class Effusion]
"""

import sys
import os
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.predictor import Predictor
from ml.inference.explainability import GradCAMExplainer


def main():
    parser = argparse.ArgumentParser(description="NIH ChestX-ray14 Grad-CAM Explainer CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input chest X-ray image file")
    parser.add_argument("--class", dest="target_class", type=str, default=None, help="Target pathology name for Grad-CAM")
    parser.add_argument("--output-dir", type=str, default="docs/phase_7_visualizations", help="Folder to save output heatmaps")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional model checkpoint path override")
    parser.add_argument("--device", type=str, default=None, help="Optional device override ('cpu' or 'cuda')")

    args = parser.parse_args()

    predictor = Predictor(checkpoint_path=args.checkpoint, device=args.device)
    explainer = GradCAMExplainer(predictor)

    result = explainer.explain(args.image, target_class=args.target_class, output_dir=args.output_dir)

    print("==================================================")
    print("NIH CHESTX-RAY14 GRAD-CAM EXPLAINABILITY RESULT")
    print("==================================================")
    print(f"Target Pathology:    {result['target_class']}")
    print(f"Pathology Prob:      {result['target_probability'] * 100:.2f}%")
    print(f"Original Image:      {result['saved_paths'].get('original_path')}")
    print(f"Grad-CAM Heatmap:    {result['saved_paths'].get('heatmap_path')}")
    print(f"Color Overlay:       {result['saved_paths'].get('overlay_path')}")
    print("--------------------------------------------------")
    print(f"{result['disclaimer']}")
    print("==================================================")


if __name__ == "__main__":
    main()
