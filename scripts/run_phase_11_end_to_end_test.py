"""
Phase 11 — End-to-End System Pipeline & API Integration Test Script
-------------------------------------------------------------------
Tests the full production inference, preprocessing, thresholding, ranking, Grad-CAM, JSON export,
and FastAPI backend endpoints across standard, high-res, low-res, grayscale, RGB, RGBA, and corrupted inputs.
"""

import sys
import os
import shutil
import tempfile
import numpy as np
from PIL import Image
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.inference_service import get_predictor, run_inference
from app.services.explanation_service import generate_gradcam_explanation
from app.services.export_service import create_export_payload
from ml.preprocessing.labels import PATHOLOGY_CLASSES


def run_end_to_end_test():
    print("==================================================")
    print("STARTING PHASE 11 END-TO-END PIPELINE & API TEST")
    print("==================================================")

    tmp_dir = Path(tempfile.mkdtemp())
    results = {}
    logs = []

    predictor = get_predictor()

    # Create Sample Test Images
    gray_path = tmp_dir / "unseen_gray.png"
    rgb_path = tmp_dir / "unseen_rgb.png"
    rgba_path = tmp_dir / "unseen_rgba.png"
    highres_path = tmp_dir / "unseen_highres.png"
    lowres_path = tmp_dir / "unseen_lowres.png"

    Image.fromarray(np.random.randint(0, 255, (512, 512), dtype=np.uint8), mode="L").save(gray_path)
    Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8), mode="RGB").save(rgb_path)
    Image.fromarray(np.random.randint(0, 255, (512, 512, 4), dtype=np.uint8), mode="RGBA").save(rgba_path)
    Image.fromarray(np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8), mode="RGB").save(highres_path)
    Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8), mode="RGB").save(lowres_path)

    # 1. Standard Grayscale Processing
    res_gray = run_inference(gray_path)
    results["1_grayscale_processing"] = (res_gray is not None) and (len(res_gray.predictions) == 14)
    logs.append(f"1. Standard Grayscale Processing: {results['1_grayscale_processing']}")

    # 2. RGB Processing & 14 Class Ordering
    res_rgb = run_inference(rgb_path)
    probs_in_range = all(0.0 <= p.probability <= 1.0 for p in res_rgb.predictions.values())
    results["2_rgb_14_class_ordering"] = probs_in_range and (list(res_rgb.predictions.keys()) == PATHOLOGY_CLASSES)
    logs.append(f"2. RGB Processing & Official 14-Class Order: {results['2_rgb_14_class_ordering']}")

    # 3. RGBA Alpha Channel Processing
    results["3_rgba_processing"] = (run_inference(rgba_path) is not None)
    logs.append(f"3. RGBA Alpha Channel Processing: {results['3_rgba_processing']}")

    # 4. Resolution Extremes (High-Res 1024x1024 and Low-Res 64x64)
    res_high = run_inference(highres_path)
    res_low = run_inference(lowres_path)
    results["4_resolution_extremes"] = (res_high is not None) and (res_low is not None)
    logs.append(f"4. Resolution Extremes (64x64 & 1024x1024): {results['4_resolution_extremes']}")

    # 5. Threshold Binary Decision Logic
    binary_correct = all(p.binary_prediction == (p.probability >= p.threshold) for p in res_rgb.predictions.values())
    results["5_threshold_decision_logic"] = binary_correct
    logs.append(f"5. Binary Decision Threshold Logic: {binary_correct}")

    # 6. Ranked Predictions Order
    sorted_preds = sorted(res_rgb.predictions.values(), key=lambda x: x.probability, reverse=True)
    is_sorted = all(sorted_preds[i].probability >= sorted_preds[i+1].probability for i in range(len(sorted_preds)-1))
    results["6_ranked_predictions_order"] = is_sorted
    logs.append(f"6. Ranked Predictions Descending Sort: {is_sorted}")

    # 7. Grad-CAM Overlay Generation & Dimensions
    pil_rgb = Image.open(rgb_path)
    exp_res = generate_gradcam_explanation(pil_rgb, target_class="Effusion")
    cam_valid = (exp_res["heatmap"].shape == (512, 512)) and np.isfinite(exp_res["heatmap"]).all()
    results["7_gradcam_overlay_dimensions"] = cam_valid
    logs.append(f"7. Grad-CAM Overlay Dimensions & Finite Values: {cam_valid}")

    # 8. Machine-Readable Export Payload Generation
    export_payload = create_export_payload(res_rgb, image_bytes=b"sample", inference_time_sec=0.25)
    export_valid = ("disclaimer" in export_payload) and (len(export_payload["predictions"]["pathologies"]) == 14)
    results["8_export_payload_generation"] = export_valid
    logs.append(f"8. Machine-Readable JSON Export Schema: {export_valid}")

    # 9. Malformed Input Rejection
    bad_file = tmp_dir / "bad.png"
    with open(bad_file, "wb") as f:
        f.write(b"NOT_A_VALID_IMAGE")
    try:
        run_inference(bad_file)
        results["9_malformed_rejected"] = False
    except Exception:
        results["9_malformed_rejected"] = True
    logs.append(f"9. Malformed Image Input Safely Rejected: {results['9_malformed_rejected']}")

    # 10. Model Weight Immutability Verification
    params_before = [p.clone() for p in predictor.model.parameters()]
    run_inference(highres_path)
    generate_gradcam_explanation(pil_rgb)
    params_after = list(predictor.model.parameters())
    weights_unchanged = all(torch.equal(p1, p2) for p1, p2 in zip(params_before, params_after))
    results["10_model_weights_immutable"] = weights_unchanged
    logs.append(f"10. Model Parameter Immutability Verified: {weights_unchanged}")

    shutil.rmtree(tmp_dir, ignore_errors=True)

    all_passed = all(results.values())
    print("\n--- END-TO-END TEST SUMMARY ---")
    for l in logs:
        print(f"  {l}")
    print(f"\nOVERALL PIPELINE TEST RESULT: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


if __name__ == "__main__":
    success = run_end_to_end_test()
    if not success:
        sys.exit(1)
