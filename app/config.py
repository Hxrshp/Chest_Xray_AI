"""
NIH ChestX-ray14 Application Configuration & Constants
-------------------------------------------------------
Centralized configuration file holding model paths, threshold settings, disclaimers, and metadata.
"""

import os
from pathlib import Path
from ml.preprocessing.labels import PATHOLOGY_CLASSES, NUM_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model & Threshold Paths
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "phase6" / "final" / "best.pth"
if not CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "phase4" / "best.pth"

THRESHOLD_PATH = PROJECT_ROOT / "data" / "processed" / "phase_5_validation_thresholds.json"

# UI Metadata & Header Titles
APP_TITLE = "Chest X-ray AI — Research Prototype"
APP_SUBTITLE = "14-Class Multi-Label Chest Radiograph Analysis & Visual Explainability"
MEDICAL_DISCLAIMER_SHORT = "RESEARCH USE ONLY — NOT FOR CLINICAL DIAGNOSIS OR PATIENT CARE"

MEDICAL_DISCLAIMER_FULL = (
    "IMPORTANT MEDICAL SAFETY DISCLAIMER: This application is an experimental research model for "
    "chest radiograph analysis. It is NOT a clinically validated diagnostic device, certified medical software, "
    "or a replacement for a qualified radiologist. Model probabilities and Grad-CAM heatmaps are statistical "
    "research outputs and must NEVER be used for primary patient diagnosis, automated triage, or clinical decision-making."
)

GRADCAM_DISCLAIMER = (
    "ATTENTION VISUALIZATION DISCLAIMER: The Grad-CAM heatmap represents model feature activation "
    "regions (model attention) and does NOT prove the presence or exact boundary of a pathological lesion."
)

# Benchmark Model Evaluation Metrics (Phase 6 Selected Baseline)
MODEL_METRICS = {
    "architecture": "DenseNet-121",
    "parameters": "6,968,206",
    "input_resolution": "320 x 320",
    "selected_experiment": "exp_008_capped_weights",
    "val_macro_auroc": "0.8352",
    "test_macro_auroc": "0.8256",
    "test_micro_auroc": "0.8524",
    "test_macro_auprc": "0.3012",
    "test_micro_auprc": "0.3418",
    "test_macro_f1": "0.3214",
    "test_micro_f1": "0.4182",
    "test_macro_brier": "0.0512",
    "test_macro_ece": "0.0384",
    "ci_95_macro_auroc": "[0.8211, 0.8299]"
}
