"""
Phase 7C — Batch Inference CLI Tool
-----------------------------------
Executes batch radiograph classification for large image sets, recording successes and failures safely.

Usage:
    python scripts/predict_batch.py --manifest data/processed/manifests/val.csv --output batch_results.json
"""

import sys
import os
import argparse
import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.predictor import Predictor


def main():
    parser = argparse.ArgumentParser(description="NIH ChestX-ray14 Batch Inference CLI")
    parser.add_argument("--manifest", type=str, required=True, help="CSV manifest file containing image paths or image_index column")
    parser.add_argument("--images-dir", type=str, default="data/raw/images", help="Root folder for raw images")
    parser.add_argument("--output", type=str, required=True, help="Output JSON filepath for batch predictions")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional model checkpoint path override")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Optional device override ('cpu' or 'cuda')")

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: Manifest file missing at {manifest_path}")
        sys.exit(1)

    df = pd.read_csv(manifest_path)
    images_dir = Path(args.images_dir)

    image_paths = []
    for idx, row in df.iterrows():
        if "image_path" in row and pd.notna(row["image_path"]):
            image_paths.append(Path(row["image_path"]))
        elif "image_index" in row:
            image_paths.append(images_dir / str(row["image_index"]))
        elif "Image Index" in row:
            image_paths.append(images_dir / str(row["Image Index"]))

    print("==================================================")
    print("NIH CHESTX-RAY14 BATCH INFERENCE PROCESSOR")
    print("==================================================")
    print(f"Total Images Requested: {len(image_paths):,}")
    print(f"Batch Size:             {args.batch_size}")

    predictor = Predictor(checkpoint_path=args.checkpoint, device=args.device)
    batch_result = predictor.predict_batch(image_paths, batch_size=args.batch_size)

    print("\n--- BATCH INFERENCE SUMMARY ---")
    print(f"Successful Predictions: {batch_result.successful_count:,}")
    print(f"Failed Image Count:     {batch_result.failed_count:,}")
    print(f"Device Used:            {batch_result.device}")

    out_file = Path(args.output)
    os.makedirs(out_file.parent, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(batch_result.to_json(indent=2))

    print(f"Saved batch prediction JSON to {out_file}")
    print("==================================================")


if __name__ == "__main__":
    main()
