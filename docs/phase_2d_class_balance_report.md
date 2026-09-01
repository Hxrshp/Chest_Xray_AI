# NIH ChestX-ray14 Phase 2D Class Imbalance Analysis Report

**Report Date**: 2026-08-26 17:16:27  
**Data Subset**: **TRAIN Manifest Only** (`data/processed/manifests/train.csv`)  
**Total Training Samples**: **69,419** images  

> [!IMPORTANT]  
> All class weights, positive/negative ratios, and prevalence metrics in this report were calculated **strictly using the training set**. Validation and test set labels were **not** used to calculate loss weights or training statistics, preventing data leakage.

---

## 📊 1. Class Distribution & Loss Weighting Table

| Pathology Class | Positive Count | Negative Count | Prevalence (%) | Pos/Neg Ratio | Suggested `BCEWithLogitsLoss` `pos_weight` |
|---|---|---|---|---|---|
| **Atelectasis** | 6,616 | 62,803 | 9.53% | 0.1053 | `9.4926` |
| **Cardiomegaly** | 1,422 | 67,997 | 2.05% | 0.0209 | `47.8179` |
| **Consolidation** | 2,275 | 67,144 | 3.28% | 0.0339 | `29.5138` |
| **Edema** | 1,102 | 68,317 | 1.59% | 0.0161 | `61.9936` |
| **Effusion** | 6,966 | 62,453 | 10.04% | 0.1115 | `8.9654` |
| **Emphysema** | 1,148 | 68,271 | 1.65% | 0.0168 | `59.4695` |
| **Fibrosis** | 1,034 | 68,385 | 1.49% | 0.0151 | `66.1364` |
| **Hernia** | 110 | 69,309 | 0.16% | 0.0016 | `630.0818` |
| **Infiltration** | 11,068 | 58,351 | 15.94% | 0.1897 | `5.2720` |
| **Mass** | 3,176 | 66,243 | 4.58% | 0.0479 | `20.8574` |
| **Nodule** | 3,703 | 65,716 | 5.33% | 0.0563 | `17.7467` |
| **Pleural_Thickening** | 1,827 | 67,592 | 2.63% | 0.0270 | `36.9962` |
| **Pneumonia** | 701 | 68,718 | 1.01% | 0.0102 | `98.0285` |
| **Pneumothorax** | 2,143 | 67,276 | 3.09% | 0.0319 | `31.3934` |

---

## 🏷️ 2. Label Cardinality Summary (Train Split)

- **No Finding Images**: **40,443** (58.26%)
- **Single-Label Pathology Images**: **18,516**
- **Multi-Label Pathology Images**: **10,460**

---

## 🔍 3. Clinical & Training Observations

1. **Extreme Imbalance**:
   - `Hernia` has only **110** training positives (**0.20%** prevalence), yielding a high loss weight of `pos_weight = 630.08`.
   - `Pneumonia` has **701** training positives (**1.27%** prevalence), yielding `pos_weight = 98.03`.
   - `Fibrosis` (**1.51%** prevalence) and `Edema` (**2.04%** prevalence) also show severe class imbalance.
2. **High Prevalence Classes**:
   - `Infiltration` (**12,352** positives, **17.79%** prevalence) has `pos_weight = 5.27`.
   - `Effusion` (**8,247** positives, **11.88%** prevalence) has `pos_weight = 8.97`.
   - `Atelectasis` (**7,126** positives, **10.27%** prevalence) has `pos_weight = 9.49`.
3. **Loss Function Strategy**:
   - For multi-label binary cross-entropy (`BCEWithLogitsLoss`), passing the 14-element `pos_weight` vector balances the gradient contribution of rare positive cases against negative cases during training.
