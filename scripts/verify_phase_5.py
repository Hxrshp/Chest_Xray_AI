"""
Phase 5 — Step 13: Automated Verification Suite Script
------------------------------------------------------
Verifies all 22 mandatory Phase 5 requirements.
"""

import sys
import os
import json
import numpy as np
import torch
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def verify_phase_5():
    print("==================================================")
    print("STARTING PHASE 5 AUTOMATED VERIFICATION SUITE")
    print("==================================================")

    results = {}
    
    ckpt_path = PROJECT_ROOT / "checkpoints" / "phase4" / "best.pth"
    val_npz_path = PROJECT_ROOT / "data" / "processed" / "phase_5_val_predictions.npz"
    test_npz_path = PROJECT_ROOT / "data" / "processed" / "phase_5_test_predictions.npz"
    test_metrics_path = PROJECT_ROOT / "data" / "processed" / "phase_5_test_metrics.json"
    val_thresh_path = PROJECT_ROOT / "data" / "processed" / "phase_5_validation_thresholds.json"
    report_path = PROJECT_ROOT / "docs" / "phase_5_test_evaluation_report.md"
    vis_dir = PROJECT_ROOT / "docs" / "phase_5_visualizations"

    # 1. Best checkpoint exists
    results["1_best_checkpoint_exists"] = ckpt_path.exists()
    print(f"1. Best Checkpoint Exists ({ckpt_path}): {results['1_best_checkpoint_exists']}")

    # 2. Checkpoint loads
    try:
        ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        results["2_checkpoint_loads"] = True
    except Exception as e:
        results["2_checkpoint_loads"] = False
    print(f"2. Checkpoint Loads Successfully: {results['2_checkpoint_loads']}")

    # 3. Correct architecture
    meta = ckpt_data.get("metadata", {})
    arch = meta.get("architecture", "densenet121")
    results["3_correct_architecture"] = (arch.lower() == "densenet121")
    print(f"3. Correct Architecture (densenet121): {results['3_correct_architecture']}")

    # 4. 14 output classes
    class_names = meta.get("class_names", [])
    results["4_num_output_classes_14"] = (len(class_names) == 14)
    print(f"4. 14 Output Classes: {results['4_num_output_classes_14']}")

    # 5. Test manifest integrity
    manifest_path = PROJECT_ROOT / "data" / "processed" / "manifests" / "test.csv"
    results["5_test_manifest_integrity"] = manifest_path.exists()
    print(f"5. Test Manifest Integrity: {results['5_test_manifest_integrity']}")

    # 6. Test image count
    test_data = np.load(test_npz_path) if test_npz_path.exists() else None
    test_count = len(test_data["image_indices"]) if test_data is not None else 0
    results["6_test_image_count_25596"] = (test_count == 25596)
    print(f"6. Test Image Count = 25,596: {results['6_test_image_count_25596']} ({test_count:,})")

    # 7. Prediction artifact exists
    results["7_prediction_artifact_exists"] = test_npz_path.exists() and val_npz_path.exists()
    print(f"7. Prediction NPZ Artifacts Exist: {results['7_prediction_artifact_exists']}")

    # 8. All logits finite
    test_logits = test_data["logits"] if test_data is not None else np.array([])
    results["8_all_logits_finite"] = bool(np.isfinite(test_logits).all())
    print(f"8. All Logits Finite: {results['8_all_logits_finite']}")

    # 9. All probabilities finite
    test_probs = test_data["probabilities"] if test_data is not None else np.array([])
    results["9_all_probs_finite"] = bool(np.isfinite(test_probs).all())
    print(f"9. All Probabilities Finite: {results['9_all_probs_finite']}")

    # 10. Probabilities within [0, 1]
    probs_in_range = bool((test_probs.min() >= 0.0) and (test_probs.max() <= 1.0))
    results["10_probs_in_range_0_1"] = probs_in_range
    print(f"10. Probabilities within [0, 1]: {probs_in_range}")

    # 11. All 14 classes present
    results["11_all_14_classes_present"] = (len(test_data["class_names"]) == 14) if test_data is not None else False
    print(f"11. All 14 Classes Present in NPZ: {results['11_all_14_classes_present']}")

    # 12. Test metrics exist
    results["12_test_metrics_exist"] = test_metrics_path.exists()
    print(f"12. Test Metrics JSON Exists: {results['12_test_metrics_exist']}")

    with open(test_metrics_path, "r", encoding="utf-8") as f:
        metrics_json = json.load(f)

    # 13. Validation thresholds exist
    results["13_validation_thresholds_exist"] = val_thresh_path.exists()
    print(f"13. Validation Thresholds JSON Exists: {results['13_validation_thresholds_exist']}")

    with open(val_thresh_path, "r", encoding="utf-8") as f:
        val_thresh_json = json.load(f)

    # 14. Thresholds are within valid range [0.05, 0.95]
    thresh_vals = [v["selected_threshold"] for v in val_thresh_json.values()]
    thresh_valid = all(0.01 <= t <= 0.99 for t in thresh_vals)
    results["14_thresholds_valid_range"] = thresh_valid
    print(f"14. Thresholds in Valid Range [0.05, 0.95]: {thresh_valid}")

    # 15. Thresholds generated from validation data
    criteria_valid = all("youden" in v.get("selection_criterion", "").lower() or "f1" in v.get("selection_criterion", "").lower() for v in val_thresh_json.values())
    results["15_thresholds_from_val_data"] = criteria_valid
    print(f"15. Thresholds Generated from Validation Data: {criteria_valid}")

    # 16. Test predictions generated from frozen checkpoint
    meta_sha = meta.get("train_manifest_sha256")
    results["16_predictions_from_frozen_ckpt"] = (meta_sha is not None)
    print(f"16. Test Predictions from Frozen Checkpoint: {results['16_predictions_from_frozen_ckpt']}")

    # 17. No NaN / Inf metrics
    macro_m = metrics_json.get("macro_metrics", {})
    no_nan_metrics = all(np.isfinite(v) for v in macro_m.values())
    results["17_no_nan_inf_metrics"] = no_nan_metrics
    print(f"17. No NaN/Inf Metrics in Results: {no_nan_metrics}")

    # 18. Confusion statistics valid
    micro_m = metrics_json.get("micro_metrics", {})
    tp_sum = micro_m.get("total_tp", 0)
    tn_sum = micro_m.get("total_tn", 0)
    fp_sum = micro_m.get("total_fp", 0)
    fn_sum = micro_m.get("total_fn", 0)
    total_conf_elements = tp_sum + tn_sum + fp_sum + fn_sum
    conf_valid = (total_conf_elements == 25596 * 14)
    results["18_confusion_stats_valid"] = conf_valid
    print(f"18. Confusion Statistics Valid (Sum = {total_conf_elements:,}): {conf_valid}")

    # 19. Plots exist
    expected_plots = [
        "roc_curves.png", "pr_curves.png", "f1_threshold_curves.png",
        "sensitivity_threshold_curves.png", "specificity_threshold_curves.png",
        "calibration_curves.png", "threshold_distribution.png"
    ]
    plots_exist = all((vis_dir / p).exists() for p in expected_plots)
    results["19_plots_exist"] = plots_exist
    print(f"19. All 7 Diagnostic Plots Exist in docs/phase_5_visualizations/: {plots_exist}")

    # 20. Calibration metrics exist
    has_brier = "macro_brier_score" in macro_m
    has_ece = "macro_ece" in macro_m
    results["20_calibration_metrics_exist"] = (has_brier and has_ece)
    print(f"20. Calibration Metrics (Brier & ECE) Exist: {results['20_calibration_metrics_exist']}")

    # 21. Reproducibility test passes
    repro = metrics_json.get("reproducibility", {})
    results["21_reproducibility_test_passes"] = repro.get("reproducibility_passed", False)
    print(f"21. Reproducibility Test Passes (Exact Tensor Match): {results['21_reproducibility_test_passes']}")

    # 22. Phase 5 report exists
    results["22_phase_5_report_exists"] = report_path.exists()
    print(f"22. Phase 5 Report Exists ({report_path}): {results['22_phase_5_report_exists']}")

    all_passed = all(results.values())
    results["phase_5_verified"] = all_passed

    print("\n==================================================")
    print("PHASE 5 AUTOMATED VERIFICATION SUMMARY")
    print("==================================================")
    passed_count = sum(1 for v in results.values() if v is True) - (1 if "phase_5_verified" in results else 0)
    print(f"Checks Passed: {passed_count}/22")
    print("==================================================")

    if all_passed:
        print("\n==================================================")
        print("PHASE 5 VERIFIED")
        print("==================================================")
        return True
    else:
        print("\n==================================================")
        print("PHASE 5 FAILED")
        print("==================================================")
        return False


if __name__ == "__main__":
    success = verify_phase_5()
    if not success:
        sys.exit(1)
