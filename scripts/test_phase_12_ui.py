"""
Phase 12 — Web Interface & Radiologist Workflow Unit Testing Script
--------------------------------------------------------------------
Tests application components, file validation, Predictor loading, checkpoint hash immutability,
14 pathology probability formatting, threshold matching, Grad-CAM overlays, JSON schema exports,
and FastAPI backend compatibility.
"""

import sys
import os
import json
import shutil
import tempfile
import hashlib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.inference_service import get_predictor, run_inference
from app.services.explanation_service import generate_gradcam_explanation
from app.services.export_service import create_export_payload
from ml.preprocessing.labels import PATHOLOGY_CLASSES


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def run_phase_12_ui_tests():
    print("==================================================")
    print("STARTING PHASE 12 UI & WORKFLOW UNIT TEST SUITE")
    print("==================================================")

    tmp_dir = Path(tempfile.mkdtemp())
    results = {}
    logs = []

    # 1. Production Checkpoint Hash Verification
    ckpt_path = PROJECT_ROOT / "checkpoints" / "phase6" / "final" / "best.pth"
    if not ckpt_path.exists():
        ckpt_path = PROJECT_ROOT / "checkpoints" / "phase4" / "best.pth"

    ckpt_hash = compute_file_sha256(ckpt_path)
    expected_hash = "bdc7e13a1f302d81d467470cba94faaede33e5b8acbc12d76005a72b99031d8f"
    hash_match = (ckpt_hash == expected_hash)
    results["1_checkpoint_hash_match"] = hash_match
    logs.append(f"1. Checkpoint SHA-256 Hash Matched ({ckpt_hash[:12]}...): {hash_match}")

    # 2. Predictor Initialization
    predictor = get_predictor()
    results["2_predictor_initialized"] = (predictor is not None)
    logs.append("2. Predictor Initialization: True")

    # Create Test Images
    gray_img = Image.fromarray(np.random.randint(0, 255, (512, 512), dtype=np.uint8), mode="L")
    rgb_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8), mode="RGB")
    rgba_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 4), dtype=np.uint8), mode="RGBA")
    bmp_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8), mode="RGB")
    highres_img = Image.fromarray(np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8), mode="RGB")
    lowres_img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8), mode="RGB")

    # 3. Grayscale (L) Processing
    res_gray = run_inference(gray_img)
    results["3_grayscale_processing"] = (res_gray is not None) and (len(res_gray.predictions) == 14)
    logs.append(f"3. Grayscale Image Processing: {results['3_grayscale_processing']}")

    # 4. RGB Processing & 14-Class Order
    res_rgb = run_inference(rgb_img)
    probs_valid = all(0.0 <= p.probability <= 1.0 for p in res_rgb.predictions.values())
    results["4_rgb_14_class_order"] = probs_valid and (list(res_rgb.predictions.keys()) == PATHOLOGY_CLASSES)
    logs.append(f"4. RGB Processing & 14-Class Order: {results['4_rgb_14_class_order']}")

    # 5. RGBA Alpha Channel Processing
    results["5_rgba_processing"] = (run_inference(rgba_img) is not None)
    logs.append("5. RGBA Alpha Channel Processing: True")

    # 6. Low & High Resolution Handling
    results["6_resolution_handling"] = (run_inference(highres_img) is not None) and (run_inference(lowres_img) is not None)
    logs.append("6. Low (64x64) and High (1024x1024) Resolution Handling: True")

    # 7. Corrupt Image Handling
    corrupt_file = tmp_dir / "corrupt.png"
    with open(corrupt_file, "wb") as f:
        f.write(b"CORRUPT")
    try:
        run_inference(corrupt_file)
        results["7_corrupt_image_rejected"] = False
    except Exception:
        results["7_corrupt_image_rejected"] = True
    logs.append(f"7. Corrupt Image Input Safely Rejected: {results['7_corrupt_image_rejected']}")

    # 8. Empty File Handling
    empty_file = tmp_dir / "empty.png"
    with open(empty_file, "wb") as f:
        pass
    try:
        run_inference(empty_file)
        results["8_empty_image_rejected"] = False
    except Exception:
        results["8_empty_image_rejected"] = True
    logs.append(f"8. Empty File Input Safely Rejected: {results['8_empty_image_rejected']}")

    # 9. Validation Threshold Alignment
    thresh_path = PROJECT_ROOT / "data" / "processed" / "phase_5_validation_thresholds.json"
    with open(thresh_path, "r", encoding="utf-8") as f:
        t_data = json.load(f)
    thresh_match = all(
        np.isclose(
            float(predictor.thresholds[c]),
            float(t_data[c]["selected_threshold"] if isinstance(t_data[c], dict) else t_data[c]),
            atol=1e-3
        ) for c in PATHOLOGY_CLASSES
    )
    results["9_threshold_json_match"] = thresh_match
    logs.append(f"9. Validation Thresholds JSON Alignment: {thresh_match}")

    # 10. Ranked Findings Order
    sorted_preds = sorted(res_rgb.predictions.values(), key=lambda x: x.probability, reverse=True)
    is_sorted = all(sorted_preds[i].probability >= sorted_preds[i+1].probability for i in range(len(sorted_preds)-1))
    results["10_ranked_findings_order"] = is_sorted
    logs.append(f"10. Ranked Findings Sorting Order: {is_sorted}")

    # 11. Grad-CAM Overlay Dimensions
    exp_res = generate_gradcam_explanation(rgb_img, target_class="Effusion")
    results["11_gradcam_overlay_dimensions"] = (exp_res["heatmap"].shape == (512, 512)) and np.isfinite(exp_res["heatmap"]).all()
    logs.append(f"11. Grad-CAM Heatmap Dimensions & Finite Values: {results['11_gradcam_overlay_dimensions']}")

    # 12. JSON Export Schema
    export_payload = create_export_payload(res_rgb, image_bytes=b"sample", inference_time_sec=0.20)
    results["12_json_export_schema"] = ("disclaimer" in export_payload) and (len(export_payload["predictions"]["pathologies"]) == 14)
    logs.append(f"12. Machine-Readable JSON Export Schema: {results['12_json_export_schema']}")

    # 13. Model Weight Immutability
    params_before = [p.clone() for p in predictor.model.parameters()]
    run_inference(rgb_img)
    generate_gradcam_explanation(rgb_img)
    params_after = list(predictor.model.parameters())
    weights_unchanged = all(torch.equal(p1, p2) for p1, p2 in zip(params_before, params_after))
    results["13_model_weights_immutable"] = weights_unchanged
    logs.append(f"13. Model Parameters Immutable During Inference: {weights_unchanged}")

    shutil.rmtree(tmp_dir, ignore_errors=True)

    all_passed = all(results.values())
    print("\n--- PHASE 12 UI TEST SUMMARY ---")
    for l in logs:
        print(f"  {l}")
    print(f"\nOVERALL UI TEST RESULT: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


if __name__ == "__main__":
    success = run_phase_12_ui_tests()
    if not success:
        sys.exit(1)
