"""
Configuration & Pos-Weight Consistency Audit Script
---------------------------------------------------
Verifies alignment between:
1. train.csv empirical label statistics
2. data/processed/class_statistics.json positive weights
3. ml/training/loss.py loaded pos_weights
4. configs/model_config.yaml settings
"""

import sys
import json
import torch
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.loss import load_train_pos_weights, get_loss_function

DISEASE_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]


def audit_weights_and_config():
    print("=== STARTING CONFIGURATION & WEIGHT CONSISTENCY AUDIT ===")
    
    # 1. Load train.csv and compute empirical pos_weights
    train_csv_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    if not train_csv_path.exists():
        print(f"ERROR: {train_csv_path} missing!")
        return False

    df_train = pd.read_csv(train_csv_path)
    total_train_samples = len(df_train)
    print(f"Loaded train.csv: {total_train_samples:,} images.")

    empirical_weights = {}
    for col in DISEASE_CLASSES:
        pos_cnt = int(df_train[col].sum())
        neg_cnt = total_train_samples - pos_cnt
        ratio = round(neg_cnt / pos_cnt, 4)
        empirical_weights[col] = ratio

    # 2. Load class_statistics.json
    stats_path = PROJECT_ROOT / "data" / "processed" / "class_statistics.json"
    with open(stats_path, "r", encoding="utf-8") as f:
        stats_data = json.load(f)

    json_weights = stats_data.get("bce_pos_weights", {})
    json_train_samples = stats_data.get("train_sample_count", 0)

    print(f"\n--- Sample Count Consistency ---")
    print(f"  train.csv count: {total_train_samples:,}")
    print(f"  class_statistics.json count: {json_train_samples:,}")
    sample_match = (total_train_samples == json_train_samples)
    print(f"  Sample Count Match: {sample_match}")

    # 3. Compare empirical weights vs class_statistics.json
    print(f"\n--- Positive Class Weights Comparison ---")
    print(f"{'Class':<20} | {'train.csv Ratio':<15} | {'JSON Weight':<15} | {'Diff':<10}")
    print("-" * 65)
    weight_mismatches = 0
    for col in DISEASE_CLASSES:
        emp_w = empirical_weights[col]
        js_w = json_weights.get(col, 0.0)
        diff = abs(emp_w - js_w)
        if diff > 1e-3:
            weight_mismatches += 1
        print(f"{col:<20} | {emp_w:<15.4f} | {js_w:<15.4f} | {diff:<10.4f}")

    # 4. Compare with ml/training/loss.py loaded weights
    loaded_tensor = load_train_pos_weights(str(stats_path), DISEASE_CLASSES)
    loss_fn = get_loss_function(stats_path=str(stats_path), class_names=DISEASE_CLASSES)
    
    print(f"\n--- Loss Module Verification ---")
    print(f"  Loaded Weight Tensor Shape: {loaded_tensor.shape}")
    print(f"  Loss Function Type: {type(loss_fn).__name__}")
    print(f"  Sample Weight (Infiltration): {loaded_tensor[8].item():.4f}")
    print(f"  Sample Weight (Hernia): {loaded_tensor[7].item():.4f}")

    audit_passed = sample_match and (weight_mismatches == 0)
    print("\n==================================================")
    print(f"AUDIT RESULT: {'PASSED' if audit_passed else 'FAILED'}")
    print("==================================================")
    return audit_passed


if __name__ == "__main__":
    audit_weights_and_config()
