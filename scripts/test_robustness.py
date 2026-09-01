"""
Phase 7F — Robustness & Safety Battery Suite
-------------------------------------------
Tests the predictor and explainer against unusual, malformed, empty, corrupted, and extreme inputs.
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

from ml.inference.predictor import Predictor
from ml.inference.explainability import GradCAMExplainer


def run_robustness_tests():
    print("==================================================")
    print("STARTING PHASE 7F INPUT ROBUSTNESS & SAFETY BATTERY")
    print("==================================================")

    tmp_dir = Path(tempfile.mkdtemp())
    results = {}
    logs = []

    predictor = Predictor()
    explainer = GradCAMExplainer(predictor)

    # 1. Valid Grayscale PNG
    gray_path = tmp_dir / "test_gray.png"
    Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8), mode="L").save(gray_path)
    try:
        res_gray = predictor.predict(gray_path)
        results["1_grayscale_png"] = True
        logs.append("1. Valid Grayscale PNG: PASSED")
    except Exception as e:
        results["1_grayscale_png"] = False
        logs.append(f"1. Valid Grayscale PNG: FAILED ({e})")

    # 2. Valid RGB PNG
    rgb_path = tmp_dir / "test_rgb.png"
    Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8), mode="RGB").save(rgb_path)
    try:
        res_rgb = predictor.predict(rgb_path)
        results["2_rgb_png"] = True
        logs.append("2. Valid RGB PNG: PASSED")
    except Exception as e:
        results["2_rgb_png"] = False
        logs.append(f"2. Valid RGB PNG: FAILED ({e})")

    # 3. Valid RGBA PNG
    rgba_path = tmp_dir / "test_rgba.png"
    Image.fromarray(np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8), mode="RGBA").save(rgba_path)
    try:
        res_rgba = predictor.predict(rgba_path)
        results["3_rgba_png"] = True
        logs.append("3. Valid RGBA PNG: PASSED")
    except Exception as e:
        results["3_rgba_png"] = False
        logs.append(f"3. Valid RGBA PNG: FAILED ({e})")

    # 4. Low Resolution (64 x 64)
    low_res_path = tmp_dir / "test_lowres.png"
    Image.fromarray(np.random.randint(0, 255, (64, 64), dtype=np.uint8), mode="L").save(low_res_path)
    try:
        res_low = predictor.predict(low_res_path)
        results["4_low_resolution"] = True
        logs.append("4. Low Resolution (64x64): PASSED")
    except Exception as e:
        results["4_low_resolution"] = False
        logs.append(f"4. Low Resolution (64x64): FAILED ({e})")

    # 5. High Resolution (2048 x 2048)
    high_res_path = tmp_dir / "test_highres.png"
    Image.fromarray(np.random.randint(0, 255, (2048, 2048), dtype=np.uint8), mode="L").save(high_res_path)
    try:
        res_high = predictor.predict(high_res_path)
        results["5_high_resolution"] = True
        logs.append("5. High Resolution (2048x2048): PASSED")
    except Exception as e:
        results["5_high_resolution"] = False
        logs.append(f"5. High Resolution (2048x2048): FAILED ({e})")

    # 6. Corrupt Image File
    corrupt_path = tmp_dir / "corrupt.png"
    with open(corrupt_path, "wb") as f:
        f.write(b"NOT_AN_IMAGE_HEADER_DATA_STREAM")
    try:
        predictor.predict(corrupt_path)
        results["6_corrupt_image_handled"] = False
        logs.append("6. Corrupt Image: FAILED (Accepted invalid file)")
    except Exception:
        results["6_corrupt_image_handled"] = True
        logs.append("6. Corrupt Image: PASSED (Safely rejected)")

    # 7. Empty File (0 Bytes)
    empty_path = tmp_dir / "empty.png"
    with open(empty_path, "wb") as f:
        pass
    try:
        predictor.predict(empty_path)
        results["7_empty_file_handled"] = False
        logs.append("7. Empty File: FAILED (Accepted 0-byte file)")
    except Exception:
        results["7_empty_file_handled"] = True
        logs.append("7. Empty File: PASSED (Safely rejected)")

    # 8. Missing File
    missing_path = tmp_dir / "non_existent_file.png"
    try:
        predictor.predict(missing_path)
        results["8_missing_file_handled"] = False
        logs.append("8. Missing File: FAILED (Did not raise FileNotFoundError)")
    except FileNotFoundError:
        results["8_missing_file_handled"] = True
        logs.append("8. Missing File: PASSED (Safely caught)")
    except Exception:
        results["8_missing_file_handled"] = True
        logs.append("8. Missing File: PASSED (Safely caught)")

    # 9. Batch Processor Robustness with Mixed Valid/Corrupt Images
    mixed_batch = [rgb_path, corrupt_path, gray_path, empty_path, missing_path]
    batch_res = predictor.predict_batch(mixed_batch)
    batch_robust = (batch_res.successful_count == 2) and (batch_res.failed_count == 3)
    results["9_batch_robustness"] = batch_robust
    logs.append(f"9. Batch Processor Robustness (Success=2, Failed=3): {batch_robust}")

    # 10. Model Parameter Immutability Test
    weights_before = [p.clone() for p in predictor.model.parameters()]
    predictor.predict(rgb_path)
    explainer.explain(rgb_path)
    weights_after = list(predictor.model.parameters())
    immutability = all(torch.equal(w1, w2) for w1, w2 in zip(weights_before, weights_after))
    results["10_parameter_immutability"] = immutability
    logs.append(f"10. Model Parameter Immutability: {immutability}")

    # 11. Grad-CAM Dimension & Range Test
    exp_res = explainer.explain(high_res_path, target_class="Effusion")
    heat = exp_res["heatmap"]
    cam_valid = (heat.shape == (2048, 2048)) and np.isfinite(heat).all() and (0.0 <= heat.min() and heat.max() <= 1.0)
    results["11_gradcam_validity"] = cam_valid
    logs.append(f"11. Grad-CAM Heatmap Validity (Shape (2048,2048), Range [0,1]): {cam_valid}")

    shutil.rmtree(tmp_dir, ignore_errors=True)

    all_passed = all(results.values())
    print("\n--- ROBUSTNESS BATTERY SUMMARY ---")
    for l in logs:
        print(f"  {l}")
    print(f"\nOVERALL ROBUSTNESS RESULT: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


if __name__ == "__main__":
    success = run_robustness_tests()
    if not success:
        sys.exit(1)
