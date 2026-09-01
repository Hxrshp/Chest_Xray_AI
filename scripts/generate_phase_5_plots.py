"""
Phase 5 — Step 8: Visualization Generator Script
------------------------------------------------
Generates high-resolution diagnostic plots for ROC curves, Precision-Recall curves,
F1/Sensitivity/Specificity threshold curves, calibration reliability diagrams, and threshold distributions.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, auc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIS_DIR = PROJECT_ROOT / "docs" / "phase_5_visualizations"
os.makedirs(VIS_DIR, exist_ok=True)

# Load Test Predictions NPZ & Test Metrics JSON
test_npz_path = PROJECT_ROOT / "data" / "processed" / "phase_5_test_predictions.npz"
test_metrics_path = PROJECT_ROOT / "data" / "processed" / "phase_5_test_metrics.json"

test_data = np.load(test_npz_path)
targets = test_data["targets"]
probs = test_data["probabilities"]
class_names = test_data["class_names"]

with open(test_metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)


def generate_all_plots():
    print("=== GENERATING PHASE 5 VISUALIZATIONS ===")

    # 1. Per-Class & Macro ROC Curves
    plt.figure(figsize=(10, 8))
    for c_idx, c_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(targets[:, c_idx], probs[:, c_idx])
        c_auroc = metrics["per_class"][c_name]["auroc"]
        plt.plot(fpr, tpr, label=f"{c_name} (AUROC: {c_auroc:.4f})", alpha=0.75, linewidth=1.5)

    plt.plot([0, 1], [0, 1], "k--", label="Random Chance (0.5000)", alpha=0.6)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.title(f"NIH ChestX-ray14 — Per-Class ROC Curves (Macro AUROC: {metrics['macro_metrics']['macro_auroc']:.4f})", fontsize=12, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved roc_curves.png")

    # 2. Per-Class & Macro Precision-Recall Curves
    plt.figure(figsize=(10, 8))
    for c_idx, c_name in enumerate(class_names):
        p, r, _ = precision_recall_curve(targets[:, c_idx], probs[:, c_idx])
        c_auprc = metrics["per_class"][c_name]["auprc"]
        plt.plot(r, p, label=f"{c_name} (AUPRC: {c_auprc:.4f})", alpha=0.75, linewidth=1.5)

    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11)
    plt.title(f"NIH ChestX-ray14 — Precision-Recall Curves (Macro AUPRC: {metrics['macro_metrics']['macro_auprc']:.4f})", fontsize=12, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(VIS_DIR / "pr_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved pr_curves.png")

    # 3. F1, Sensitivity, Specificity vs Threshold Curves
    thresholds_grid = np.linspace(0.05, 0.95, 37)

    fig_f1, ax_f1 = plt.subplots(figsize=(10, 6))
    fig_sens, ax_sens = plt.subplots(figsize=(10, 6))
    fig_spec, ax_spec = plt.subplots(figsize=(10, 6))

    for c_idx, c_name in enumerate(class_names):
        f1s, sens_list, spec_list = [], [], []
        c_t = targets[:, c_idx]
        c_p = probs[:, c_idx]

        for th in thresholds_grid:
            bin_preds = (c_p >= th).astype(int)
            tp = np.sum((c_t == 1) & (bin_preds == 1))
            fp = np.sum((c_t == 0) & (bin_preds == 1))
            fn = np.sum((c_t == 1) & (bin_preds == 0))
            tn = np.sum((c_t == 0) & (bin_preds == 0))

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0

            f1s.append(f1)
            sens_list.append(sens)
            spec_list.append(spec)

        ax_f1.plot(thresholds_grid, f1s, label=c_name, alpha=0.75, linewidth=1.5)
        ax_sens.plot(thresholds_grid, sens_list, label=c_name, alpha=0.75, linewidth=1.5)
        ax_spec.plot(thresholds_grid, spec_list, label=c_name, alpha=0.75, linewidth=1.5)

    ax_f1.set_xlabel("Decision Threshold", fontsize=11)
    ax_f1.set_ylabel("F1 Score", fontsize=11)
    ax_f1.set_title("F1 Score vs Decision Threshold across 14 Pathologies", fontsize=12, fontweight="bold")
    ax_f1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax_f1.grid(True, linestyle="--", alpha=0.5)
    fig_f1.tight_layout()
    fig_f1.savefig(VIS_DIR / "f1_threshold_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig_f1)
    print("  Saved f1_threshold_curves.png")

    ax_sens.set_xlabel("Decision Threshold", fontsize=11)
    ax_sens.set_ylabel("Sensitivity (True Positive Rate)", fontsize=11)
    ax_sens.set_title("Sensitivity vs Decision Threshold across 14 Pathologies", fontsize=12, fontweight="bold")
    ax_sens.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax_sens.grid(True, linestyle="--", alpha=0.5)
    fig_sens.tight_layout()
    fig_sens.savefig(VIS_DIR / "sensitivity_threshold_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig_sens)
    print("  Saved sensitivity_threshold_curves.png")

    ax_spec.set_xlabel("Decision Threshold", fontsize=11)
    ax_spec.set_ylabel("Specificity (True Negative Rate)", fontsize=11)
    ax_spec.set_title("Specificity vs Decision Threshold across 14 Pathologies", fontsize=12, fontweight="bold")
    ax_spec.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax_spec.grid(True, linestyle="--", alpha=0.5)
    fig_spec.tight_layout()
    fig_spec.savefig(VIS_DIR / "specificity_threshold_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig_spec)
    print("  Saved specificity_threshold_curves.png")

    # 4. Calibration Reliability Diagrams
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    axes = axes.ravel()

    for c_idx, c_name in enumerate(class_names):
        ax = axes[c_idx]
        rd = metrics["per_class"][c_name]["reliability_diagram"]
        accs = rd["bin_accuracies"]
        confs = rd["bin_confidences"]
        ece = metrics["per_class"][c_name]["ece"]

        ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect Calibration")
        ax.plot(confs, accs, "s-", color="#1f77b4", label=f"ECE: {ece:.4f}")
        ax.set_title(c_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Confidence", fontsize=9)
        ax.set_ylabel("Empirical Accuracy", fontsize=9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)

    # Hide unused subplots
    for i in range(len(class_names), len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle(f"NIH ChestX-ray14 — Reliability Calibration Diagrams (Macro ECE: {metrics['macro_metrics']['macro_ece']:.4f})", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(VIS_DIR / "calibration_curves.png", dpi=300)
    plt.close(fig)
    print("  Saved calibration_curves.png")

    # 5. Threshold Distribution Bar Chart
    val_thresh_path = PROJECT_ROOT / "data" / "processed" / "phase_5_validation_thresholds.json"
    with open(val_thresh_path, "r", encoding="utf-8") as f:
        val_threshs = json.load(f)

    thresh_values = [val_threshs[c]["selected_threshold"] for c in class_names]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, thresh_values, color="#2ca02c", edgecolor="black", alpha=0.8)
    plt.axhline(0.50, color="r", linestyle="--", label="Standard Default Threshold (0.50)")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.ylabel("Validation-Derived Decision Threshold", fontsize=11)
    plt.title("Validation-Derived Optimal Threshold Distribution (Youden's J Statistic)", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f"{height:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(VIS_DIR / "threshold_distribution.png", dpi=300)
    plt.close()
    print("  Saved threshold_distribution.png")

    print("\nALL PHASE 5 VISUALIZATIONS GENERATED SUCCESSFULLY!")


if __name__ == "__main__":
    generate_all_plots()
