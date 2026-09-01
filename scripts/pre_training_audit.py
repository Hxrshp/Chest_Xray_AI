"""
Phase 4 Mandatory Pre-Training Audit Script
-------------------------------------------
Executes Section 0 (Pre-Training Audit) & Section 1 (Class-Weight Consistency Audit)
"""

import sys
import os
import json
import hashlib
import pandas as pd
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.loss import load_train_pos_weights, get_loss_function

DISEASE_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]


def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def run_pre_training_audit():
    print("==================================================")
    print("SECTION 0: MANDATORY PRE-TRAINING AUDIT")
    print("==================================================")
    
    # 0.1 Check mandatory files exist
    required_files = [
        PROJECT_ROOT / "data" / "processed" / "manifests" / "train.csv",
        PROJECT_ROOT / "data" / "processed" / "manifests" / "val.csv",
        PROJECT_ROOT / "data" / "processed" / "manifests" / "test.csv",
        PROJECT_ROOT / "data" / "processed" / "class_statistics.json",
        PROJECT_ROOT / "configs" / "data_config.yaml",
        PROJECT_ROOT / "data" / "processed" / "split_verification.json"
    ]

    all_exist = True
    for fp in required_files:
        exists = fp.exists()
        print(f"  {fp.relative_to(PROJECT_ROOT)}: {'EXISTS' if exists else 'MISSING'}")
        if not exists:
            all_exist = False
    
    if not all_exist:
        print("ERROR: Mandatory files missing!")
        return False

    # 0.2 Check SHA-256 hashes against split_verification.json
    split_ver_path = PROJECT_ROOT / "data" / "processed" / "split_verification.json"
    with open(split_ver_path, "r", encoding="utf-8") as f:
        split_ver = json.load(f)

    manifest_dir = PROJECT_ROOT / "data" / "processed" / "manifests"
    train_hash = compute_sha256(manifest_dir / "train.csv")
    val_hash = compute_sha256(manifest_dir / "val.csv")
    test_hash = compute_sha256(manifest_dir / "test.csv")

    exp_train_hash = split_ver["manifest_sha256"]["train_sha256"]
    exp_val_hash = split_ver["manifest_sha256"]["val_sha256"]
    exp_test_hash = split_ver["manifest_sha256"]["test_sha256"]

    hash_match = (train_hash == exp_train_hash) and (val_hash == exp_val_hash) and (test_hash == exp_test_hash)
    print(f"\n--- Manifest SHA-256 Hash Verification ---")
    print(f"  Train Hash Match: {train_hash == exp_train_hash} ({train_hash[:12]}...)")
    print(f"  Val Hash Match:   {val_hash == exp_val_hash} ({val_hash[:12]}...)")
    print(f"  Test Hash Match:  {test_hash == exp_test_hash} ({test_hash[:12]}...)")

    # 0.3 Verify Image & Patient Counts and Leakage
    df_train = pd.read_csv(manifest_dir / "train.csv")
    df_val = pd.read_csv(manifest_dir / "val.csv")
    df_test = pd.read_csv(manifest_dir / "test.csv")

    train_cnt, val_cnt, test_cnt = len(df_train), len(df_val), len(df_test)
    total_cnt = train_cnt + val_cnt + test_cnt

    count_valid = (train_cnt == 69419) and (val_cnt == 17105) and (test_cnt == 25596) and (total_cnt == 112120)
    print(f"\n--- Image Count Verification ---")
    print(f"  Train: {train_cnt:,} (Exp: 69,419) -> {train_cnt == 69419}")
    print(f"  Val:   {val_cnt:,} (Exp: 17,105) -> {val_cnt == 17105}")
    print(f"  Test:  {test_cnt:,} (Exp: 25,596) -> {test_cnt == 25596}")
    print(f"  Total: {total_cnt:,} (Exp: 112,120) -> {total_cnt == 112120}")

    # Patient / Image Overlap Check
    train_pts = set(df_train["patient_id"])
    val_pts = set(df_val["patient_id"])
    test_pts = set(df_test["patient_id"])

    pt_overlap = len(train_pts & val_pts) + len(train_pts & test_pts) + len(val_pts & test_pts)
    
    train_imgs = set(df_train["image_index"])
    val_imgs = set(df_val["image_index"])
    test_imgs = set(df_test["image_index"])

    img_overlap = len(train_imgs & val_imgs) + len(train_imgs & test_imgs) + len(val_imgs & test_imgs)

    print(f"\n--- Leakage Verification ---")
    print(f"  Patient Overlap: {pt_overlap} (Exp: 0)")
    print(f"  Image Overlap:   {img_overlap} (Exp: 0)")
    leakage_valid = (pt_overlap == 0) and (img_overlap == 0)

    # Image Files Directory Check
    img_dir = PROJECT_ROOT / "data" / "raw" / "images"
    actual_img_count = sum(1 for e in os.scandir(img_dir) if e.name.endswith(".png"))
    print(f"\n--- Raw Image Folder Check ---")
    print(f"  PNG Files on Disk: {actual_img_count:,} (Exp: 112,120)")
    img_files_valid = (actual_img_count == 112120)

    sec0_passed = all_exist and hash_match and count_valid and leakage_valid and img_files_valid
    print(f"\nSECTION 0 VERIFICATION: {'PASSED' if sec0_passed else 'FAILED'}")

    print("\n==================================================")
    print("SECTION 1: CLASS-WEIGHT CONSISTENCY AUDIT")
    print("==================================================")

    stats_path = PROJECT_ROOT / "data" / "processed" / "class_statistics.json"
    with open(stats_path, "r", encoding="utf-8") as f:
        stats_data = json.load(f)

    json_weights = stats_data.get("bce_pos_weights", {})
    json_stats = stats_data.get("class_statistics", {})

    print(f"{'Class':<20} | {'Pos Count':<10} | {'Neg Count':<10} | {'Prevalence %':<12} | {'Pos Weight':<12}")
    print("-" * 75)

    empirical_weights = {}
    weight_mismatches = 0

    for idx, col in enumerate(DISEASE_CLASSES):
        pos_cnt = int(df_train[col].sum())
        neg_cnt = train_cnt - pos_cnt
        prev_pct = round((pos_cnt / train_cnt) * 100, 3)
        pos_w = round(neg_cnt / pos_cnt, 4)
        empirical_weights[col] = pos_w

        js_w = json_weights.get(col, 0.0)
        js_pos = json_stats.get(col, {}).get("positive_count", 0)
        js_neg = json_stats.get(col, {}).get("negative_count", 0)

        diff = abs(pos_w - js_w)
        if diff > 1e-3 or pos_cnt != js_pos or neg_cnt != js_neg:
            weight_mismatches += 1

        print(f"{col:<20} | {pos_cnt:<10} | {neg_cnt:<10} | {prev_pct:<12.3f} | {pos_w:<12.4f}")

    # Verify ml/training/loss.py loads weights dynamically
    loaded_tensor, _ = load_train_pos_weights(str(stats_path))
    loss_fn, _ = get_loss_function({"loss": {"type": "weighted_bce", "use_pos_weight": True}}, class_stats_path=str(stats_path))

    print(f"\n--- Loss Module Verification ---")
    print(f"  Loaded Weight Tensor Shape: {loaded_tensor.shape} (Exp: [14])")
    print(f"  Loss Function Type: {type(loss_fn).__name__} (Exp: BCEWithLogitsLoss)")
    print(f"  Sample Weight (Infiltration [idx=8]): {loaded_tensor[8].item():.4f}")
    print(f"  Sample Weight (Hernia [idx=7]): {loaded_tensor[7].item():.4f}")

    # Check exact class ordering
    manifest_targets = [c for c in df_train.columns if c in DISEASE_CLASSES]
    ordering_valid = (manifest_targets == DISEASE_CLASSES) and (list(json_weights.keys()) == DISEASE_CLASSES)
    print(f"  Exact Class Ordering Valid Across All Modules: {ordering_valid}")

    sec1_passed = (weight_mismatches == 0) and ordering_valid and (loaded_tensor.shape[0] == 14)
    print(f"\nSECTION 1 VERIFICATION: {'PASSED' if sec1_passed else 'FAILED'}")

    overall_passed = sec0_passed and sec1_passed
    print("\n==================================================")
    print(f"PRE-TRAINING AUDIT OVERALL RESULT: {'PASSED' if overall_passed else 'FAILED'}")
    print("==================================================")
    return overall_passed


if __name__ == "__main__":
    success = run_pre_training_audit()
    if not success:
        sys.exit(1)
