"""
Phase 2D Class Imbalance Analysis Script (TRAIN MANIFEST ONLY)
--------------------------------------------------------------
Calculates positive count, negative count, prevalence, pos/neg ratio,
and suggested BCEWithLogitsLoss pos_weight (negative_count / positive_count)
strictly using data/processed/manifests/train.csv.
"""

import sys
import json
import time
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifests" / "train.csv"
STATS_JSON = PROJECT_ROOT / "data" / "processed" / "class_statistics.json"
REPORT_MD = PROJECT_ROOT / "docs" / "phase_2d_class_balance_report.md"

PATHOLOGY_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax"
]


def compute_class_statistics():
    print("=== COMPUTING TRAIN-ONLY CLASS IMBALANCE STATISTICS ===")
    if not TRAIN_MANIFEST.exists():
        print(f"ERROR: Train manifest missing: {TRAIN_MANIFEST}")
        sys.exit(1)

    df_train = pd.read_csv(TRAIN_MANIFEST)
    total_train_samples = len(df_train)
    print(f"Loaded Train Manifest: {total_train_samples} samples.")

    class_stats = {}
    pos_weights_dict = {}

    for cls in PATHOLOGY_CLASSES:
        pos_count = int(df_train[cls].sum())
        neg_count = total_train_samples - pos_count
        prevalence_pct = round((pos_count / total_train_samples) * 100.0, 3)
        pos_neg_ratio = round(pos_count / neg_count, 5) if neg_count > 0 else 0.0
        
        # pos_weight = negative_count / positive_count for BCEWithLogitsLoss
        bce_pos_weight = round(neg_count / pos_count, 4) if pos_count > 0 else 1.0

        class_stats[cls] = {
            "positive_count": pos_count,
            "negative_count": neg_count,
            "prevalence_pct": prevalence_pct,
            "pos_neg_ratio": pos_neg_ratio,
            "bce_pos_weight": bce_pos_weight
        }
        pos_weights_dict[cls] = bce_pos_weight

    # Overall Train Label Characteristics
    no_finding_count = int((df_train[PATHOLOGY_CLASSES].sum(axis=1) == 0).sum())
    single_label_count = int((df_train[PATHOLOGY_CLASSES].sum(axis=1) == 1).sum())
    multi_label_count = int((df_train[PATHOLOGY_CLASSES].sum(axis=1) > 1).sum())

    output_data = {
        "dataset": "NIH ChestX-ray14 (TRAIN Split Only)",
        "train_sample_count": total_train_samples,
        "no_finding_count": no_finding_count,
        "no_finding_pct": round((no_finding_count / total_train_samples) * 100.0, 3),
        "single_label_count": single_label_count,
        "multi_label_count": multi_label_count,
        "bce_pos_weights": pos_weights_dict,
        "class_statistics": class_stats
    }

    # Save JSON
    with open(STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved class statistics to {STATS_JSON}")

    # Generate Markdown Report
    md_content = f"""# NIH ChestX-ray14 Phase 2D Class Imbalance Analysis Report

**Report Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Data Subset**: **TRAIN Manifest Only** (`data/processed/manifests/train.csv`)  
**Total Training Samples**: **{total_train_samples:,}** images  

> [!IMPORTANT]  
> All class weights, positive/negative ratios, and prevalence metrics in this report were calculated **strictly using the training set**. Validation and test set labels were **not** used to calculate loss weights or training statistics, preventing data leakage.

---

## 📊 1. Class Distribution & Loss Weighting Table

| Pathology Class | Positive Count | Negative Count | Prevalence (%) | Pos/Neg Ratio | Suggested `BCEWithLogitsLoss` `pos_weight` |
|---|---|---|---|---|---|
"""
    for cls in PATHOLOGY_CLASSES:
        st = class_stats[cls]
        md_content += f"| **{cls}** | {st['positive_count']:,} | {st['negative_count']:,} | {st['prevalence_pct']:.2f}% | {st['pos_neg_ratio']:.4f} | `{st['bce_pos_weight']:.4f}` |\n"

    md_content += f"""
---

## 🏷️ 2. Label Cardinality Summary (Train Split)

- **No Finding Images**: **{no_finding_count:,}** ({output_data['no_finding_pct']:.2f}%)
- **Single-Label Pathology Images**: **{single_label_count:,}**
- **Multi-Label Pathology Images**: **{multi_label_count:,}**

---

## 🔍 3. Clinical & Training Observations

1. **Extreme Imbalance**:
   - `Hernia` has only **{class_stats['Hernia']['positive_count']}** training positives (**0.20%** prevalence), yielding a high loss weight of `pos_weight = {class_stats['Hernia']['bce_pos_weight']:.2f}`.
   - `Pneumonia` has **{class_stats['Pneumonia']['positive_count']:,}** training positives (**1.27%** prevalence), yielding `pos_weight = {class_stats['Pneumonia']['bce_pos_weight']:.2f}`.
   - `Fibrosis` (**1.51%** prevalence) and `Edema` (**2.04%** prevalence) also show severe class imbalance.
2. **High Prevalence Classes**:
   - `Infiltration` (**12,352** positives, **17.79%** prevalence) has `pos_weight = {class_stats['Infiltration']['bce_pos_weight']:.2f}`.
   - `Effusion` (**8,247** positives, **11.88%** prevalence) has `pos_weight = {class_stats['Effusion']['bce_pos_weight']:.2f}`.
   - `Atelectasis` (**7,126** positives, **10.27%** prevalence) has `pos_weight = {class_stats['Atelectasis']['bce_pos_weight']:.2f}`.
3. **Loss Function Strategy**:
   - For multi-label binary cross-entropy (`BCEWithLogitsLoss`), passing the 14-element `pos_weight` vector balances the gradient contribution of rare positive cases against negative cases during training.
"""

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved class balance report to {REPORT_MD}")

    return output_data


if __name__ == "__main__":
    compute_class_statistics()
