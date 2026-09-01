"""
Phase 5 — Complete Baseline Test-Set Evaluation, Threshold Optimization, Calibration & Error Analysis Script
-----------------------------------------------------------------------------------------------------------------
Executes frozen inference on Val & Test sets, optimizes thresholds on Val set ONLY, applies fixed thresholds to Test set,
calculates discrimination/calibration metrics, confusion statistics, and reproducibility.
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.builder import build_model
from ml.training.checkpointing import load_checkpoint
from ml.preprocessing.loaders import get_dataloaders
from ml.preprocessing.labels import PATHOLOGY_CLASSES


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10):
    """
    Computes Expected Calibration Error (ECE) for binary predictions.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(probs)

    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (probs > bin_lower) & (probs <= bin_upper) if i > 0 else (probs >= bin_lower) & (probs <= bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            accuracy = np.mean(targets[in_bin])
            confidence = np.mean(probs[in_bin])
            bin_weight = bin_count / total_samples
            ece += bin_weight * np.abs(accuracy - confidence)
            bin_accs.append(float(accuracy))
            bin_confs.append(float(confidence))
            bin_counts.append(int(bin_count))
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            bin_counts.append(0)

    return ece, bin_accs, bin_confs, bin_counts


def run_inference(model, dataloader, device):
    """
    Runs frozen deterministic inference over dataloader and returns labels, logits, probabilities, and filenames.
    """
    model.eval()
    all_logits = []
    all_targets = []
    image_indices = []

    with torch.no_grad():
        for batch in dataloader:
            images, targets = batch[0], batch[1]
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(images)

            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets.numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_probs = 1.0 / (1.0 + np.exp(-all_logits))

    # Fetch filenames from dataset
    dataset_df = dataloader.dataset.df
    image_indices = dataset_df["image_index"].values

    return image_indices, all_targets, all_logits, all_probs


def execute_phase_5():
    print("==================================================")
    print("STARTING PHASE 5 — TEST-SET EVALUATION & ANALYSIS")
    print("==================================================")
    start_time = time.time()

    # 1. Load Configurations
    with open(PROJECT_ROOT / "configs" / "model_config.yaml", "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    with open(PROJECT_ROOT / "configs" / "data_config.yaml", "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on Device: {device} (CUDA Available: {torch.cuda.is_available()})")

    # 2. DataLoaders (Validation and Test Sets)
    print("\n--- Initializing Validation & Test DataLoaders ---")
    _, val_loader, test_loader = get_dataloaders(data_cfg)
    print(f"  Validation Set: {len(val_loader.dataset):,} images")
    print(f"  Test Set:       {len(test_loader.dataset):,} images")

    # 3. Load Best Model Checkpoint
    best_path = PROJECT_ROOT / "checkpoints" / "phase4" / "best.pth"
    print(f"\n--- Loading Best Checkpoint: {best_path} ---")
    model = build_model(model_cfg).to(device)
    ckpt_meta = load_checkpoint(str(best_path), model=model, device=device)
    print(f"  Loaded Epoch: {ckpt_meta.get('epoch', 'N/A')}")
    print(f"  Recorded Best Val Macro AUROC: {ckpt_meta.get('val_macro_auroc', 0.0):.4f}")

    # 4. Step 3 — Frozen Inference on Validation Set
    print("\n--- Running Frozen Inference on VALIDATION Set (17,105 images) ---")
    val_indices, val_targets, val_logits, val_probs = run_inference(model, val_loader, device)

    # Save Val Predictions NPZ
    val_npz_path = PROJECT_ROOT / "data" / "processed" / "phase_5_val_predictions.npz"
    np.savez_compressed(
        val_npz_path,
        image_indices=val_indices,
        targets=val_targets,
        logits=val_logits,
        probabilities=val_probs,
        class_names=np.array(PATHOLOGY_CLASSES)
    )
    print(f"Saved validation predictions to {val_npz_path}")

    # 5. Step 3 — Frozen Inference on TEST Set
    print("\n--- Running Frozen Inference on TEST Set (25,596 images) ---")
    test_indices, test_targets, test_logits, test_probs = run_inference(model, test_loader, device)

    # Save Test Predictions NPZ
    test_npz_path = PROJECT_ROOT / "data" / "processed" / "phase_5_test_predictions.npz"
    np.savez_compressed(
        test_npz_path,
        image_indices=test_indices,
        targets=test_targets,
        logits=test_logits,
        probabilities=test_probs,
        class_names=np.array(PATHOLOGY_CLASSES)
    )
    print(f"Saved test predictions to {test_npz_path}")

    # 6. Step 5 — Validation-Only Threshold Optimization
    print("\n--- Step 5: Validation-Only Threshold Optimization ---")
    candidate_thresholds = np.linspace(0.05, 0.95, 19)
    val_thresholds = {}

    for c_idx, class_name in enumerate(PATHOLOGY_CLASSES):
        c_targets = val_targets[:, c_idx]
        c_probs = val_probs[:, c_idx]

        best_j = -1.0
        best_j_thresh = 0.5
        best_f1 = -1.0
        best_f1_thresh = 0.5

        for thresh in candidate_thresholds:
            preds = (c_probs >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(c_targets, preds, labels=[0, 1]).ravel()

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
            j_stat = sens + spec - 1.0

            if j_stat > best_j:
                best_j = j_stat
                best_j_thresh = float(thresh)

            if f1 > best_f1:
                best_f1 = f1
                best_f1_thresh = float(thresh)

        # Primary selection criterion: Youden's J statistic
        val_thresholds[class_name] = {
            "class_index": c_idx,
            "youden_j_threshold": round(best_j_thresh, 4),
            "youden_j_val_score": round(best_j, 4),
            "f1_optimal_threshold": round(best_f1_thresh, 4),
            "f1_val_score": round(best_f1, 4),
            "selected_threshold": round(best_j_thresh, 4),
            "selection_criterion": "Youden_J_statistic"
        }

    val_thresh_json_path = PROJECT_ROOT / "data" / "processed" / "phase_5_validation_thresholds.json"
    with open(val_thresh_json_path, "w", encoding="utf-8") as f:
        json.dump(val_thresholds, f, indent=2)
    print(f"Saved validation thresholds to {val_thresh_json_path}")

    # 7. Step 4 & 6 — Test-Set Metrics Calculation & Fixed Threshold Application
    print("\n--- Step 4 & 6: Test-Set Metric Evaluation & Thresholding ---")
    test_metrics = {
        "per_class": {},
        "macro_metrics": {},
        "micro_metrics": {},
        "calibration": {}
    }

    per_class_aurocs = []
    per_class_auprcs = []
    per_class_aps = []
    per_class_sens = []
    per_class_spec = []
    per_class_prec = []
    per_class_f1s = []
    per_class_briers = []
    per_class_eces = []

    all_test_tp = 0
    all_test_tn = 0
    all_test_fp = 0
    all_test_fn = 0

    for c_idx, class_name in enumerate(PATHOLOGY_CLASSES):
        c_targets = test_targets[:, c_idx]
        c_probs = test_probs[:, c_idx]
        fixed_thresh = val_thresholds[class_name]["selected_threshold"]

        # Threshold-independent metrics
        c_auroc = float(roc_auc_score(c_targets, c_probs))
        c_ap = float(average_precision_score(c_targets, c_probs))

        prec_arr, rec_arr, _ = precision_recall_curve(c_targets, c_probs)
        c_auprc = float(auc(rec_arr, prec_arr))

        # Threshold-dependent metrics
        preds = (c_probs >= fixed_thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(c_targets, preds, labels=[0, 1]).ravel()

        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = float(2 * (prec * sens) / (prec + sens)) if (prec + sens) > 0 else 0.0

        # Calibration metrics
        brier = float(brier_score_loss(c_targets, c_probs))
        ece, b_accs, b_confs, b_counts = compute_ece(c_probs, c_targets, n_bins=10)

        per_class_aurocs.append(c_auroc)
        per_class_auprcs.append(c_auprc)
        per_class_aps.append(c_ap)
        per_class_sens.append(sens)
        per_class_spec.append(spec)
        per_class_prec.append(prec)
        per_class_f1s.append(f1)
        per_class_briers.append(brier)
        per_class_eces.append(ece)

        all_test_tp += int(tp)
        all_test_tn += int(tn)
        all_test_fp += int(fp)
        all_test_fn += int(fn)

        pos_cnt = int(np.sum(c_targets))
        neg_cnt = int(len(c_targets) - pos_cnt)
        prev_pct = float((pos_cnt / len(c_targets)) * 100)

        test_metrics["per_class"][class_name] = {
            "prevalence_pct": round(prev_pct, 4),
            "pos_count": pos_cnt,
            "neg_count": neg_cnt,
            "auroc": round(c_auroc, 4),
            "auprc": round(c_auprc, 4),
            "average_precision": round(c_ap, 4),
            "validation_threshold": fixed_thresh,
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "precision": round(prec, 4),
            "f1_score": round(f1, 4),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "brier_score": round(brier, 4),
            "ece": round(ece, 4),
            "reliability_diagram": {
                "bin_accuracies": b_accs,
                "bin_confidences": b_confs,
                "bin_counts": b_counts
            }
        }

    # Micro Metrics
    flat_test_targets = test_targets.ravel()
    flat_test_probs = test_probs.ravel()
    micro_auroc = float(roc_auc_score(flat_test_targets, flat_test_probs))
    micro_ap = float(average_precision_score(flat_test_targets, flat_test_probs))

    p_micro, r_micro, _ = precision_recall_curve(flat_test_targets, flat_test_probs)
    micro_auprc = float(auc(r_micro, p_micro))

    micro_sens = all_test_tp / (all_test_tp + all_test_fn) if (all_test_tp + all_test_fn) > 0 else 0.0
    micro_spec = all_test_tn / (all_test_tn + all_test_fp) if (all_test_tn + all_test_fp) > 0 else 0.0
    micro_prec = all_test_tp / (all_test_tp + all_test_fp) if (all_test_tp + all_test_fp) > 0 else 0.0
    micro_f1 = 2 * (micro_prec * micro_sens) / (micro_prec + micro_sens) if (micro_prec + micro_sens) > 0 else 0.0

    test_metrics["macro_metrics"] = {
        "macro_auroc": round(float(np.mean(per_class_aurocs)), 4),
        "macro_auprc": round(float(np.mean(per_class_auprcs)), 4),
        "macro_average_precision": round(float(np.mean(per_class_aps)), 4),
        "macro_sensitivity": round(float(np.mean(per_class_sens)), 4),
        "macro_specificity": round(float(np.mean(per_class_spec)), 4),
        "macro_precision": round(float(np.mean(per_class_prec)), 4),
        "macro_f1": round(float(np.mean(per_class_f1s)), 4),
        "macro_brier_score": round(float(np.mean(per_class_briers)), 4),
        "macro_ece": round(float(np.mean(per_class_eces)), 4)
    }

    test_metrics["micro_metrics"] = {
        "micro_auroc": round(micro_auroc, 4),
        "micro_auprc": round(micro_auprc, 4),
        "micro_average_precision": round(micro_ap, 4),
        "micro_sensitivity": round(micro_sens, 4),
        "micro_specificity": round(micro_spec, 4),
        "micro_precision": round(micro_prec, 4),
        "micro_f1": round(micro_f1, 4),
        "total_tp": all_test_tp,
        "total_tn": all_test_tn,
        "total_fp": all_test_fp,
        "total_fn": all_test_fn
    }

    test_metrics_path = PROJECT_ROOT / "data" / "processed" / "phase_5_test_metrics.json"
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Saved test metrics to {test_metrics_path}")

    # 8. Step 12 — Reproducibility Test
    print("\n--- Step 12: Test-Set Inference Reproducibility Test ---")
    _, test_targets_2, test_logits_2, test_probs_2 = run_inference(model, test_loader, device)

    logits_close = np.allclose(test_logits, test_logits_2, rtol=0, atol=0)
    probs_close = np.allclose(test_probs, test_probs_2, rtol=0, atol=0)
    print(f"  Run 1 vs Run 2 Logits Exact Match (allclose rtol=0, atol=0): {logits_close}")
    print(f"  Run 1 vs Run 2 Probs Exact Match (allclose rtol=0, atol=0):  {probs_close}")

    test_metrics["reproducibility"] = {
        "logits_exact_match": bool(logits_close),
        "probs_exact_match": bool(probs_close),
        "reproducibility_passed": bool(logits_close and probs_close)
    }

    # Re-save updated json
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n==================================================")
    print(f"PHASE 5 INFERENCE & EVALUATION COMPLETED IN {elapsed:.1f}s")
    print(f"  Test Macro AUROC:  {test_metrics['macro_metrics']['macro_auroc']}")
    print(f"  Test Micro AUROC:  {test_metrics['micro_metrics']['micro_auroc']}")
    print(f"  Test Macro AUPRC:  {test_metrics['macro_metrics']['macro_auprc']}")
    print(f"  Test Micro AUPRC:  {test_metrics['micro_metrics']['micro_auprc']}")
    print(f"  Test Macro F1:     {test_metrics['macro_metrics']['macro_f1']}")
    print(f"==================================================")


if __name__ == "__main__":
    execute_phase_5()
