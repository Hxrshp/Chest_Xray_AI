"""
Phase 7B — Single Image Inference CLI Tool
------------------------------------------
Executes single-image radiograph classification returning raw logits, probabilities, and binary predictions.

Usage:
    python scripts/predict_image.py --image data/raw/images/00000001_000.png [--output result.json]
"""

import sys
import os
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.predictor import Predictor
from ml.preprocessing.labels import PATHOLOGY_CLASSES


def main():
    parser = argparse.ArgumentParser(description="NIH ChestX-ray14 Single Image Predictor CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input chest X-ray image file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional model checkpoint path override")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON filepath")
    parser.add_argument("--device", type=str, default=None, help="Optional device override ('cpu' or 'cuda')")

    args = parser.parse_args()

    predictor = Predictor(checkpoint_path=args.checkpoint, device=args.device)
    result = predictor.predict(args.image)

    print("==================================================")
    print("NIH CHESTX-RAY14 SINGLE IMAGE INFERENCE RESULT")
    print("==================================================")
    print(f"Image File:          {result.image_path}")
    print(f"Highest Pathology:   {result.highest_probability_class} ({result.highest_probability * 100:.2f}%)")
    print(f"Inference Device:    {result.device}")
    print(f"Checkpoint Used:     {result.model_checkpoint}")
    print("--------------------------------------------------")
    print(f"{'Pathology':<22} | {'Logit':<8} | {'Prob %':<8} | {'Thresh':<7} | {'Decision'}")
    print("--------------------------------------------------")

    for c_name in PATHOLOGY_CLASSES:
        p = result.predictions[c_name]
        decision = "POSITIVE" if p.binary_prediction else "Negative"
        thresh_str = f"{p.threshold:.2f}" if p.threshold is not None else "N/A"
        print(f"{c_name:<22} | {p.raw_logit:<8.4f} | {p.probability * 100:<8.2f} | {thresh_str:<7} | {decision}")

    print("==================================================")
    print(f"\n{result.disclaimer}\n")

    if args.output:
        out_file = Path(args.output)
        os.makedirs(out_file.parent, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(result.to_json(indent=2))
        print(f"Saved machine-readable prediction JSON to {out_file}")


if __name__ == "__main__":
    main()
