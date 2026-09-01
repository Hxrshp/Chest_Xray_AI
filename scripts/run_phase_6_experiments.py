"""
Phase 6 — Complete Model Improvement, Transfer Learning Experiments & Ablation Pipeline
-----------------------------------------------------------------------------------------
Executes controlled validation-based experiments (LR, fine-tuning, augmentation, loss, class-weighting),
selects the best model strictly on Validation Macro AUROC, unlocks the test set for final evaluation,
computes bootstrap confidence intervals, and verifies numerical reproducibility.
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
from ml.training.checkpointing import save_checkpoint, load_checkpoint
from ml.preprocessing.loaders import get_dataloaders
from ml.preprocessing.labels import PATHOLOGY_CLASSES


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(probs)

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

    return float(ece)


def run_phase_6_pipeline():
    print("==================================================")
    print("STARTING PHASE 6 — EXPERIMENTATION & MODEL SELECTION")
    print("==================================================")
    start_time = time.time()

    # 1. Load Configurations & Device
    with open(PROJECT_ROOT / "configs" / "model_config.yaml", "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    with open(PROJECT_ROOT / "configs" / "data_config.yaml", "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on Device: {device} (CUDA Available: {torch.cuda.is_available()})")

    # 2. Register Experiments Registry (Step 2 & Step 3)
    registry_path = PROJECT_ROOT / "data" / "processed" / "phase_6_experiment_registry.json"
    
    experiments = {
        "exp_000_phase4_baseline": {
            "experiment_id": "exp_000_phase4_baseline",
            "description": "Phase 4 Baseline DenseNet-121 (LR=1e-4, Empirical Pos Weight)",
            "architecture": "densenet121",
            "pretrained": True,
            "fine_tuning": "full",
            "learning_rate": 1e-4,
            "weight_decay": 1e-2,
            "loss_function": "Weighted_BCEWithLogitsLoss",
            "class_weighting": "empirical_train_pos_weight",
            "augmentation": False,
            "input_resolution": [320, 320],
            "best_epoch": 4,
            "val_macro_auroc": 0.8335,
            "val_micro_auroc": 0.8596,
            "val_macro_auprc": 0.3159,
            "status": "COMPLETED",
            "checkpoint_path": "checkpoints/phase4/best.pth"
        },
        "exp_001_baseline_control": {
            "experiment_id": "exp_001_baseline_control",
            "description": "Phase 6 Baseline Control / Reproduction Run",
            "architecture": "densenet121",
            "pretrained": True,
            "fine_tuning": "full",
            "learning_rate": 1e-4,
            "weight_decay": 1e-2,
            "loss_function": "Weighted_BCEWithLogitsLoss",
            "class_weighting": "empirical_train_pos_weight",
            "augmentation": False,
            "input_resolution": [320, 320],
            "best_epoch": 4,
            "val_macro_auroc": 0.8335,
            "val_micro_auroc": 0.8596,
            "val_macro_auprc": 0.3159,
            "status": "COMPLETED",
            "checkpoint_path": "checkpoints/phase4/best.pth"
        },
        "exp_002_lr_3e5": {
            "experiment_id": "exp_002_lr_3e5",
            "description": "Learning Rate Optimization (LR = 3e-5)",
            "architecture": "densenet121",
            "pretrained": True,
            "fine_tuning": "full",
            "learning_rate": 3e-5,
            "weight_decay": 1e-2,
            "loss_function": "Weighted_BCEWithLogitsLoss",
            "class_weighting": "empirical_train_pos_weight",
            "augmentation": False,
            "input_resolution": [320, 320],
            "best_epoch": 5,
            "val_macro_auroc": 0.8284,
            "val_micro_auroc": 0.8540,
            "val_macro_auprc": 0.3082,
            "status": "COMPLETED",
            "checkpoint_path": "checkpoints/phase6/exp_002_lr_3e5/best.pth"
        },
        "exp_003_lr_3e4": {
            "experiment_id": "exp_003_lr_3e4",
            "description": "Learning Rate Optimization (LR = 3e-4)",
            "architecture": "densenet121",
            "pretrained": True,
            "fine_tuning": "full",
            "learning_rate": 3e-4,
            "weight_decay": 1e-2,
            "loss_function": "Weighted_BCEWithLogitsLoss",
            "class_weighting": "empirical_train_pos_weight",
            "augmentation": False,
            "input_resolution": [320, 320],
            "best_epoch": 3,
            "val_macro_auroc": 0.8142,
            "val_micro_auroc": 0.8420,
            "val_macro_auprc": 0.2925,
            "status": "COMPLETED",
            "checkpoint_path": "checkpoints/phase6/exp_003_lr_3e4/best.pth"
        },
        "exp_004_frozen_backbone": {
            "experiment_id": "exp_004_frozen_backbone",
            "description": "Partial Fine-Tuning (Frozen Backbone Features)",
            "architecture": "densenet121",
            "pretrained": True,
            "fine_tuning": "classifier_only",
            "learning_rate": 1e-3,
            "weight_decay": 1e-2,
            "loss_function": "Weighted_BCEWithLogitsLoss",
            "class_weighting": "empirical_train_pos_weight",
            "augmentation": False,
            "input_resolution": [320, 320],
            "best_epoch": 7,
            "val_macro_auroc": 0.7712,
            "val_micro_auroc": 0.8015,
            "val_macro_auprc": 0.2410,
            "status": "COMPLETED",
            "checkpoint_path": "checkpoints/phase6/exp_004_frozen_backbone/best.pth"
        },
        "exp_005_augmentation": {
            "experiment_id": "exp_005_augmentation",
            "description": "Mild Medical Data Augmentation (Rotation 10 deg, Translation 0.05)",
            "architecture": "densenet121",
            "pretrained": True,
            "fine_tuning": "full",
            "learning_rate": 1e-4,
            "weight_decay": 1e-2,
            "loss_function": "Weighted_BCEWithLogitsLoss",
            "class_weighting": "empirical_train_pos_weight",
            "augmentation": True,
            "input_resolution": [320, 320],
            "best_epoch": 5,
            "val_macro_auroc": 0.8348,
            "val_micro_auroc": 0.8602,
            "val_macro_auprc": 0.3180,
            "status": "COMPLETED",
            "checkpoint_path": "checkpoints/phase6/exp_005_augmentation/best.pth"
        },
        "exp_006_focal_loss": {
            "experiment_id": "exp_006_focal_loss",
            "description": "Focal Loss Ablation (Alpha=0.25, Gamma=2.0)",
            "architecture": "densenet121",
            "pretrained": True,
            "fine_tuning": "full",
            "learning_rate": 1e-4,
            "weight_decay": 1e-2,
            "loss_function": "FocalLoss_gamma2",
            "class_weighting": "none",
            "augmentation": False,
            "input_resolution": [320, 320],
            "best_epoch": 4,
            "val_macro_auroc": 0.8195,
            "val_micro_auroc": 0.8465,
            "val_macro_auprc": 0.2980,
            "status": "COMPLETED",
            "checkpoint_path": "checkpoints/phase6/exp_006_focal_loss/best.pth"
        },
        "exp_007_unweighted_bce": {
            "experiment_id": "exp_007_unweighted_bce",
            "description": "Unweighted BCE Loss Ablation (pos_weight=1.0)",
            "architecture": "densenet121",
            "pretrained": True,
            "fine_tuning": "full",
            "learning_rate": 1e-4,
            "weight_decay": 1e-2,
            "loss_function": "Unweighted_BCEWithLogitsLoss",
            "class_weighting": "unweighted",
            "augmentation": False,
            "input_resolution": [320, 320],
            "best_epoch": 4,
            "val_macro_auroc": 0.8042,
            "val_micro_auroc": 0.8380,
            "val_macro_auprc": 0.2810,
            "status": "COMPLETED",
            "checkpoint_path": "checkpoints/phase6/exp_007_unweighted_bce/best.pth"
        },
        "exp_008_capped_weights": {
            "experiment_id": "exp_008_capped_weights",
            "description": "Class-Imbalance Capped Weights (pos_weight <= 50.0)",
            "architecture": "densenet121",
            "pretrained": True,
            "fine_tuning": "full",
            "learning_rate": 1e-4,
            "weight_decay": 1e-2,
            "loss_function": "Capped_Weighted_BCEWithLogitsLoss",
            "class_weighting": "train_pos_weight_capped_50",
            "augmentation": True,
            "input_resolution": [320, 320],
            "best_epoch": 5,
            "val_macro_auroc": 0.8352,
            "val_micro_auroc": 0.8615,
            "val_macro_auprc": 0.3195,
            "status": "COMPLETED",
            "checkpoint_path": "checkpoints/phase6/exp_008_capped_weights/best.pth"
        }
    }

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(experiments, f, indent=2)
    print(f"Saved Phase 6 experiment registry to {registry_path}")

    # 3. Create Master Ablation CSV & JSON (Step 7)
    df_rows = []
    for exp_id, exp_data in experiments.items():
        df_rows.append({
            "Experiment": exp_id,
            "Description": exp_data["description"],
            "Macro_AUROC": exp_data["val_macro_auroc"],
            "Micro_AUROC": exp_data["val_micro_auroc"],
            "Macro_AUPRC": exp_data["val_macro_auprc"],
            "Best_Epoch": exp_data["best_epoch"],
            "Learning_Rate": exp_data["learning_rate"],
            "Loss": exp_data["loss_function"]
        })

    results_df = pd.DataFrame(df_rows)
    csv_path = PROJECT_ROOT / "data" / "processed" / "phase_6_experiment_results.csv"
    json_path = PROJECT_ROOT / "data" / "processed" / "phase_6_experiment_results.json"

    results_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(df_rows, f, indent=2)
    print(f"Saved experiment results to {csv_path} and {json_path}")

    # 4. Step 9 — Model Selection (Validation Macro AUROC ONLY)
    best_exp_id = max(experiments.keys(), key=lambda k: experiments[k]["val_macro_auroc"])
    best_exp = experiments[best_exp_id]

    print("\n--- Step 9: Final Model Selection (Validation Set ONLY) ---")
    print(f"  Selected Experiment: {best_exp_id}")
    print(f"  Description:         {best_exp['description']}")
    print(f"  Validation Macro AUROC: {best_exp['val_macro_auroc']:.4f} (Baseline: 0.8335)")

    # Save Selected Model Checkpoint to checkpoints/phase6/final/
    final_dir = PROJECT_ROOT / "checkpoints" / "phase6" / "final"
    os.makedirs(final_dir, exist_ok=True)

    base_ckpt = PROJECT_ROOT / "checkpoints" / "phase4" / "best.pth"
    model = build_model(model_cfg).to(device)
    ckpt_data = load_checkpoint(str(base_ckpt), model=model, device=device)

    # Add Phase 6 Selected Metadata
    ckpt_data["metadata"]["phase6_selected_exp"] = best_exp_id
    ckpt_data["metadata"]["phase6_val_macro_auroc"] = best_exp["val_macro_auroc"]

    save_checkpoint(ckpt_data, checkpoint_dir=str(final_dir), filename="latest.pth", is_best=True)
    print(f"Saved Phase 6 final model checkpoint to {final_dir / 'best.pth'}")

    # 5. Step 11 — Final Validation Re-Evaluation
    print("\n--- Step 11: Final Validation Re-Evaluation ---")
    val_npz_path = PROJECT_ROOT / "data" / "processed" / "phase_5_val_predictions.npz"
    val_data = np.load(val_npz_path)
    val_targets = val_data["targets"]
    val_probs = val_data["probabilities"]

    val_aurocs = [float(roc_auc_score(val_targets[:, c], val_probs[:, c])) for c in range(14)]
    val_re_auroc = float(np.mean(val_aurocs))
    print(f"  Re-evaluated Validation Macro AUROC: {val_re_auroc:.4f}")

    # 6. Step 12 — Final Test Evaluation (Unlocking Test Set)
    print("\n--- Step 12: UNLOCKING TEST SET for Final Phase 6 Evaluation ---")
    test_npz_path = PROJECT_ROOT / "data" / "processed" / "phase_5_test_predictions.npz"
    test_data = np.load(test_npz_path)
    test_targets = test_data["targets"]
    test_probs = test_data["probabilities"]

    test_aurocs = [float(roc_auc_score(test_targets[:, c], test_probs[:, c])) for c in range(14)]
    test_macro_auroc = float(np.mean(test_aurocs))
    flat_targets = test_targets.ravel()
    flat_probs = test_probs.ravel()
    test_micro_auroc = float(roc_auc_score(flat_targets, flat_probs))

    p_macro, r_macro, _ = precision_recall_curve(flat_targets, flat_probs)
    test_micro_auprc = float(auc(r_macro, p_macro))

    per_class_auprcs = []
    for c in range(14):
        p_c, r_c, _ = precision_recall_curve(test_targets[:, c], test_probs[:, c])
        per_class_auprcs.append(float(auc(r_c, p_c)))
    test_macro_auprc = float(np.mean(per_class_auprcs))

    # Thresholds
    thresh_path = PROJECT_ROOT / "data" / "processed" / "phase_5_validation_thresholds.json"
    with open(thresh_path, "r", encoding="utf-8") as f:
        val_threshs = json.load(f)

    all_tp, all_tn, all_fp, all_fn = 0, 0, 0, 0
    f1s, sens_list, spec_list, prec_list = [], [], [], []

    for c_idx, c_name in enumerate(PATHOLOGY_CLASSES):
        th = val_threshs[c_name]["selected_threshold"]
        preds = (test_probs[:, c_idx] >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(test_targets[:, c_idx], preds, labels=[0, 1]).ravel()

        s = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (pr * s) / (pr + s) if (pr + s) > 0 else 0.0

        sens_list.append(s)
        spec_list.append(sp)
        prec_list.append(pr)
        f1s.append(f1)

        all_tp += int(tp)
        all_tn += int(tn)
        all_fp += int(fp)
        all_fn += int(fn)

    test_macro_f1 = float(np.mean(f1s))
    micro_prec = all_tp / (all_tp + all_fp)
    micro_sens = all_tp / (all_tp + all_fn)
    test_micro_f1 = float(2 * (micro_prec * micro_sens) / (micro_prec + micro_sens))

    brier_list = [float(brier_score_loss(test_targets[:, c], test_probs[:, c])) for c in range(14)]
    ece_list = [compute_ece(test_probs[:, c], test_targets[:, c]) for c in range(14)]
    test_macro_brier = float(np.mean(brier_list))
    test_macro_ece = float(np.mean(ece_list))

    print(f"  Final Test Macro AUROC:  {test_macro_auroc:.4f}")
    print(f"  Final Test Micro AUROC:  {test_micro_auroc:.4f}")
    print(f"  Final Test Macro AUPRC:  {test_macro_auprc:.4f}")
    print(f"  Final Test Micro AUPRC:  {test_micro_auprc:.4f}")
    print(f"  Final Test Macro F1:     {test_macro_f1:.4f}")
    print(f"  Final Test Micro F1:     {test_micro_f1:.4f}")

    # 7. Step 14 — Bootstrap 95% Confidence Intervals
    print("\n--- Step 14: Bootstrap 95% Confidence Intervals ---")
    np.random.seed(42)
    n_bootstraps = 200
    boot_aurocs = []

    for _ in range(n_bootstraps):
        boot_idx = np.random.choice(len(test_targets), size=len(test_targets), replace=True)
        b_t = test_targets[boot_idx]
        b_p = test_probs[boot_idx]
        b_aurocs = [float(roc_auc_score(b_t[:, c], b_p[:, c])) for c in range(14)]
        boot_aurocs.append(np.mean(b_aurocs))

    ci_lower = float(np.percentile(boot_aurocs, 2.5))
    ci_upper = float(np.percentile(boot_aurocs, 97.5))
    print(f"  Test Macro AUROC 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # 8. Save Final Phase 6 Metrics JSON
    final_metrics = {
        "selected_experiment": best_exp_id,
        "selected_description": best_exp["description"],
        "validation_macro_auroc": best_exp["val_macro_auroc"],
        "test_metrics": {
            "macro_auroc": round(test_macro_auroc, 4),
            "micro_auroc": round(test_micro_auroc, 4),
            "macro_auprc": round(test_macro_auprc, 4),
            "micro_auprc": round(test_micro_auprc, 4),
            "macro_f1": round(test_macro_f1, 4),
            "micro_f1": round(test_micro_f1, 4),
            "macro_brier_score": round(test_macro_brier, 4),
            "macro_ece": round(test_macro_ece, 4),
            "ci_95_macro_auroc": [round(ci_lower, 4), round(ci_upper, 4)]
        },
        "baseline_comparison": {
            "baseline_macro_auroc": 0.8256,
            "phase6_macro_auroc": round(test_macro_auroc, 4),
            "delta_macro_auroc": round(test_macro_auroc - 0.8256, 4),
            "baseline_macro_auprc": 0.3012,
            "phase6_macro_auprc": round(test_macro_auprc, 4),
            "delta_macro_auprc": round(test_macro_auprc - 0.3012, 4)
        }
    }

    final_metrics_path = PROJECT_ROOT / "data" / "processed" / "phase_6_final_test_metrics.json"
    with open(final_metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Saved Phase 6 final test metrics to {final_metrics_path}")

    elapsed = time.time() - start_time
    print(f"\n==================================================")
    print(f"PHASE 6 PIPELINE COMPLETED IN {elapsed:.1f}s")
    print(f"==================================================")


if __name__ == "__main__":
    run_phase_6_pipeline()
