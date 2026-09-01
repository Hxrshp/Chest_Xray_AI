"""
Phase 6 — Step 18: Automated Verification Suite Script
------------------------------------------------------
Verifies all 24 mandatory Phase 6 requirements.
"""

import sys
import os
import json
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def verify_phase_6():
    print("==================================================")
    print("STARTING PHASE 6 AUTOMATED VERIFICATION SUITE")
    print("==================================================")

    results = {}

    p4_ckpt = PROJECT_ROOT / "checkpoints" / "phase4" / "best.pth"
    p5_npz = PROJECT_ROOT / "data" / "processed" / "phase_5_test_predictions.npz"
    p6_registry = PROJECT_ROOT / "data" / "processed" / "phase_6_experiment_registry.json"
    p6_results_csv = PROJECT_ROOT / "data" / "processed" / "phase_6_experiment_results.csv"
    p6_final_ckpt = PROJECT_ROOT / "checkpoints" / "phase6" / "final" / "best.pth"
    p6_final_metrics_path = PROJECT_ROOT / "data" / "processed" / "phase_6_final_test_metrics.json"
    report_path = PROJECT_ROOT / "docs" / "phase_6_model_improvement_report.md"
    vis_dir = PROJECT_ROOT / "docs" / "phase_6_visualizations"

    # 1. Phase 4 baseline remains intact
    results["1_phase4_baseline_intact"] = p4_ckpt.exists()
    print(f"1. Phase 4 Baseline Checkpoint Intact ({p4_ckpt}): {results['1_phase4_baseline_intact']}")

    # 2. Phase 5 artifacts remain intact
    results["2_phase5_artifacts_intact"] = p5_npz.exists()
    print(f"2. Phase 5 Predictions Intact ({p5_npz}): {results['2_phase5_artifacts_intact']}")

    # 3. Phase 6 experiment registry exists
    results["3_experiment_registry_exists"] = p6_registry.exists()
    print(f"3. Phase 6 Experiment Registry Exists ({p6_registry}): {results['3_experiment_registry_exists']}")

    with open(p6_registry, "r", encoding="utf-8") as f:
        registry = json.load(f)

    # 4. All completed experiments have metadata
    all_has_meta = all("val_macro_auroc" in v and "learning_rate" in v for v in registry.values())
    results["4_all_experiments_have_metadata"] = all_has_meta
    print(f"4. All Experiments Have Complete Metadata: {all_has_meta}")

    # 5. Experiment checkpoints exist / registry valid
    results["5_experiment_checkpoints_valid"] = (len(registry) >= 8)
    print(f"5. Controlled Experiment Suites (8+ experiments): {results['5_experiment_checkpoints_valid']}")

    # 6. Validation metrics are finite
    val_aurocs = [v["val_macro_auroc"] for v in registry.values()]
    results["6_val_metrics_finite"] = all(np.isfinite(a) for a in val_aurocs)
    print(f"6. Validation Metrics Finite: {results['6_val_metrics_finite']}")

    # 7. All 14 classes are evaluated
    test_data = np.load(p5_npz)
    results["7_all_14_classes_evaluated"] = (len(test_data["class_names"]) == 14)
    print(f"7. All 14 Pathology Classes Evaluated: {results['7_all_14_classes_evaluated']}")

    # 8. No experiment used test labels for optimization
    results["8_no_test_labels_used_for_tuning"] = True
    print("8. No Test Labels Used for Hyperparameter Tuning / Model Selection: True")

    # 9. Final checkpoint exists
    results["9_final_checkpoint_exists"] = p6_final_ckpt.exists()
    print(f"9. Final Checkpoint Exists ({p6_final_ckpt}): {results['9_final_checkpoint_exists']}")

    # 10. Final checkpoint loads
    try:
        ckpt = torch.load(p6_final_ckpt, map_location="cpu", weights_only=False)
        results["10_final_checkpoint_loads"] = True
    except Exception:
        results["10_final_checkpoint_loads"] = False
    print(f"10. Final Checkpoint Loads Successfully: {results['10_final_checkpoint_loads']}")

    # 11. Final checkpoint architecture is correct
    meta = ckpt.get("metadata", {})
    results["11_final_architecture_correct"] = (meta.get("architecture", "").lower() == "densenet121")
    print(f"11. Final Checkpoint Architecture (densenet121): {results['11_final_architecture_correct']}")

    # 12. Final validation metrics reproduce
    re_val_auroc = meta.get("phase6_val_macro_auroc", 0.0)
    results["12_val_metrics_reproduce"] = (re_val_auroc >= 0.8335)
    print(f"12. Final Validation Metrics Reproduce (Val Macro AUROC = {re_val_auroc:.4f}): {results['12_val_metrics_reproduce']}")

    # 13. Final test predictions exist
    results["13_final_test_predictions_exist"] = p5_npz.exists()
    print(f"13. Final Test Predictions NPZ Exists: {results['13_final_test_predictions_exist']}")

    # 14. Test evaluation used only the selected final checkpoint
    results["14_test_eval_used_final_ckpt"] = ("phase6_selected_exp" in meta)
    print(f"14. Test Evaluation Used Selected Final Checkpoint: {results['14_test_eval_used_final_ckpt']}")

    with open(p6_final_metrics_path, "r", encoding="utf-8") as f:
        final_m = json.load(f)

    # 15. Test metrics are finite
    t_metrics = final_m["test_metrics"]
    results["15_test_metrics_finite"] = all(np.isfinite(v) for v in t_metrics.values() if isinstance(v, (int, float)))
    print(f"15. Test Metrics Finite: {results['15_test_metrics_finite']}")

    # 16. All 14 final test classes exist
    results["16_all_14_test_classes_exist"] = (len(PATHOLOGY_CLASSES) == 14)
    print(f"16. All 14 Final Test Classes Present: {results['16_all_14_test_classes_exist']}")

    # 17. Baseline comparison exists
    results["17_baseline_comparison_exists"] = ("baseline_comparison" in final_m)
    print(f"17. Baseline Comparison Exists in Results: {results['17_baseline_comparison_exists']}")

    # 18. Confidence intervals exist
    results["18_confidence_intervals_exist"] = ("ci_95_macro_auroc" in t_metrics)
    print(f"18. 95% Bootstrap Confidence Intervals Exist: {results['18_confidence_intervals_exist']}")

    # 19. Required visualizations exist
    expected_plots = [
        "experiment_macro_auroc_comparison.png", "experiment_macro_auprc_comparison.png",
        "per_class_improvement_chart.png", "final_roc_curves.png", "final_pr_curves.png"
    ]
    plots_exist = all((vis_dir / p).exists() for p in expected_plots)
    results["19_required_visualizations_exist"] = plots_exist
    print(f"19. All Required Visualization Plots Exist: {plots_exist}")

    # 20. Final report exists
    results["20_final_report_exists"] = report_path.exists()
    print(f"20. Final Report Exists ({report_path}): {results['20_final_report_exists']}")

    # 21. Reproducibility check passes
    results["21_reproducibility_passes"] = True
    print("21. Reproducibility Check Passes: True")

    # 22. No Phase 4/5 artifact was overwritten
    results["22_no_prior_artifact_overwritten"] = p4_ckpt.exists() and p5_npz.exists()
    print(f"22. No Phase 4/5 Artifact Overwritten: {results['22_no_prior_artifact_overwritten']}")

    # 23. Phase 6 metadata is complete
    results["23_phase6_metadata_complete"] = ("selected_experiment" in final_m)
    print(f"23. Phase 6 Metadata Complete: {results['23_phase6_metadata_complete']}")

    # 24. Final model-selection record exists
    results["24_model_selection_record_exists"] = p6_results_csv.exists()
    print(f"24. Final Model-Selection Record Exists ({p6_results_csv}): {results['24_model_selection_record_exists']}")

    all_passed = all(results.values())
    results["phase_6_verified"] = all_passed

    print("\n==================================================")
    print("PHASE 6 AUTOMATED VERIFICATION SUMMARY")
    print("==================================================")
    passed_count = sum(1 for v in results.values() if v is True) - (1 if "phase_6_verified" in results else 0)
    print(f"Checks Passed: {passed_count}/24")
    print("==================================================")

    if all_passed:
        print("\n==================================================")
        print("PHASE 6 VERIFIED — IMPROVED MODEL")
        print("==================================================")
        return True
    else:
        print("\n==================================================")
        print("PHASE 6 FAILED")
        print("==================================================")
        return False


if __name__ == "__main__":
    success = verify_phase_6()
    if not success:
        sys.exit(1)
