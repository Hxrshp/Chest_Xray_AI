"""
Multi-Label Medical Evaluation Metrics Suite
---------------------------------------------
"""

import numpy as np
import torch
from typing import Dict, Any, Union
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from ml.preprocessing.labels import PATHOLOGY_CLASSES


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid function for numpy arrays.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def evaluate_multilabel_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_logits: Union[np.ndarray, torch.Tensor],
    thresholds: Union[float, np.ndarray, list] = 0.5,
) -> Dict[str, Any]:
    """
    Evaluates comprehensive multi-label medical metrics for 14 pathology classes.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_logits, torch.Tensor):
        y_logits = y_logits.detach().cpu().numpy()

    y_probs = sigmoid(y_logits)
    num_samples, num_classes = y_true.shape

    if isinstance(thresholds, (float, int)):
        threshold_list = [float(thresholds)] * num_classes
    elif isinstance(thresholds, (list, np.ndarray)):
        threshold_list = [float(t) for t in thresholds]
    else:
        threshold_list = [0.5] * num_classes

    per_class_metrics = {}
    auroc_list = []
    auprc_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    specificity_list = []

    for idx, cls_name in enumerate(PATHOLOGY_CLASSES):
        y_t = y_true[:, idx]
        y_p = y_probs[:, idx]
        thresh = threshold_list[idx]
        y_pred = (y_p >= thresh).astype(int)

        pos_count = int(np.sum(y_t))
        neg_count = num_samples - pos_count
        prevalence = pos_count / max(num_samples, 1)

        # AUROC & AUPRC calculation
        if len(np.unique(y_t)) > 1:
            try:
                auc = float(roc_auc_score(y_t, y_p))
            except Exception:
                auc = float("nan")
            try:
                auprc = float(average_precision_score(y_t, y_p))
            except Exception:
                auprc = float("nan")
        else:
            auc = float("nan")
            auprc = float("nan")

        # Confusion Matrix calculation
        if pos_count > 0 and neg_count > 0:
            cm = confusion_matrix(y_t, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
        else:
            tp = int(np.sum((y_t == 1) & (y_pred == 1)))
            fp = int(np.sum((y_t == 0) & (y_pred == 1)))
            fn = int(np.sum((y_t == 1) & (y_pred == 0)))
            tn = int(np.sum((y_t == 0) & (y_pred == 0)))

        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0

        if not np.isnan(auc):
            auroc_list.append(auc)
        if not np.isnan(auprc):
            auprc_list.append(auprc)
        if not np.isnan(f1):
            f1_list.append(f1)
        if not np.isnan(prec):
            precision_list.append(prec)
        if not np.isnan(sens):
            recall_list.append(sens)
        if not np.isnan(spec):
            specificity_list.append(spec)

        per_class_metrics[cls_name] = {
            "auroc": auc,
            "auprc": auprc,
            "sensitivity": sens,
            "specificity": spec,
            "precision": prec,
            "f1": f1,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "prevalence": prevalence,
            "threshold": thresh,
        }

    # Macro Metrics
    macro_auroc = float(np.nanmean(auroc_list)) if len(auroc_list) > 0 else float("nan")
    macro_auprc = float(np.nanmean(auprc_list)) if len(auprc_list) > 0 else float("nan")
    macro_f1 = float(np.nanmean(f1_list)) if len(f1_list) > 0 else 0.0
    macro_sensitivity = float(np.nanmean(recall_list)) if len(recall_list) > 0 else float("nan")
    macro_specificity = float(np.nanmean(specificity_list)) if len(specificity_list) > 0 else float("nan")

    # Micro Metrics (pooled across all classes)
    try:
        micro_auroc = float(roc_auc_score(y_true.ravel(), y_probs.ravel()))
    except Exception:
        micro_auroc = float("nan")
    try:
        micro_auprc = float(average_precision_score(y_true.ravel(), y_probs.ravel()))
    except Exception:
        micro_auprc = float("nan")

    return {
        "macro_auroc": macro_auroc,
        "micro_auroc": micro_auroc,
        "macro_auprc": macro_auprc,
        "micro_auprc": micro_auprc,
        "macro_f1": macro_f1,
        "macro_sensitivity": macro_sensitivity,
        "macro_specificity": macro_specificity,
        "per_class": per_class_metrics,
    }
