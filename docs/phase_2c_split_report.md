# NIH ChestX-ray14 Phase 2C Reproducible Dataset Splitting & Audit Report

**Report Date**: 2026-08-26 17:13:37  
**Processing Time**: 25.43 seconds  
**Random Seed**: `42` (Deterministic `np.random.default_rng(42)`)  
**Ordering Canonicalization**: Lexicographical Patient ID Sorting + Canonical `image_index` CSV Export Sorting  
**Reproducibility Verification**: **100% Byte-Identical Across Runs**  

---

## 🔒 1. Manifest Cryptographic Hashes (SHA-256)

| Manifest File | Image Count | SHA-256 Checksum Hash | Reproducibility Status |
|---|---|---|---|
| `data/processed/manifests/train.csv` | 69,419 | `a3158bb7de313e876af199e1a4333bbcce26301b61677d8673b055501e2774b7` | ✅ MATCH (Byte-Identical) |
| `data/processed/manifests/val.csv` | 17,105 | `50b0eb72e7aa9322cf93afa49d4510ee211d2429083cff02bec8b173c2d6968d` | ✅ MATCH (Byte-Identical) |
| `data/processed/manifests/test.csv` | 25,596 | `ab009f326c4f6eda6d70c75cba9fc9458d9fe9ac3cd3ee057c3f4eecc5e6540c` | ✅ MATCH (Byte-Identical) |

---

## 📊 2. Dataset & Split Provenance Summary

- **Primary Source Metadata**: `data/raw/Data_Entry_2017.csv` (112,120 images)
- **Official NIH Train/Val Pool**: `data/raw/train_val_list.txt` (86,524 images)
- **Official NIH Test Set**: `data/raw/test_list.txt` (25,596 images - **UNTOUCHED**)
- **Generated Candidate Lists**: `data/raw/generated_candidate_train_list.txt` & `test_list.txt` (**EXPLICITLY UNUSED**)

| Split Category | Image Count | Patient Count | Patient Overlap | Image Overlap | Manifest File |
|---|---|---|---|---|---|
| **TRAIN** | 69,419 | 22,406 | 0 | 0 | `data/processed/manifests/train.csv` |
| **VALIDATION** | 17,105 | 5,602 | 0 | 0 | `data/processed/manifests/val.csv` |
| **TEST (Official)** | 25,596 | 2,797 | 0 | 0 | `data/processed/manifests/test.csv` |
| **TOTAL** | **112,120** | **30,805** | **0** | **0** | **100% Partitioned** |

---

## 🛡️ 3. Strict Leakage & Integrity Verification

- **Train & Val Patient Overlap**: **0** (PASSED)
- **Train & Test Patient Overlap**: **0** (PASSED)
- **Val & Test Patient Overlap**: **0** (PASSED)
- **Train & Val Image Overlap**: **0** (PASSED)
- **Train & Test Image Overlap**: **0** (PASSED)
- **Val & Test Image Overlap**: **0** (PASSED)
- **Missing / Duplicate Image Assignments**: **0** (PASSED)

---

## 📈 4. Per-Class Pathology Distribution Audit

| Pathology Class | Overall Positives (Prevalence %) | Train Positives (%) | Val Positives (%) | Test Positives (%) |
|---|---|---|---|---|
| **Atelectasis** | 11,559 (10.31%) | 6,616 (9.53%) | 1,664 (9.73%) | 3,279 (12.81%) |
| **Cardiomegaly** | 2,776 (2.48%) | 1,422 (2.05%) | 285 (1.67%) | 1,069 (4.18%) |
| **Consolidation** | 4,667 (4.16%) | 2,275 (3.28%) | 577 (3.37%) | 1,815 (7.09%) |
| **Edema** | 2,303 (2.05%) | 1,102 (1.59%) | 276 (1.61%) | 925 (3.61%) |
| **Effusion** | 13,317 (11.88%) | 6,966 (10.04%) | 1,693 (9.90%) | 4,658 (18.20%) |
| **Emphysema** | 2,516 (2.24%) | 1,148 (1.65%) | 275 (1.61%) | 1,093 (4.27%) |
| **Fibrosis** | 1,686 (1.50%) | 1,034 (1.49%) | 217 (1.27%) | 435 (1.70%) |
| **Hernia** | 227 (0.20%) | 110 (0.16%) | 31 (0.18%) | 86 (0.34%) |
| **Infiltration** | 19,894 (17.74%) | 11,068 (15.94%) | 2,714 (15.87%) | 6,112 (23.88%) |
| **Mass** | 5,782 (5.16%) | 3,176 (4.58%) | 858 (5.02%) | 1,748 (6.83%) |
| **Nodule** | 6,331 (5.65%) | 3,703 (5.33%) | 1,005 (5.88%) | 1,623 (6.34%) |
| **Pleural_Thickening** | 3,385 (3.02%) | 1,827 (2.63%) | 415 (2.43%) | 1,143 (4.47%) |
| **Pneumonia** | 1,431 (1.28%) | 701 (1.01%) | 175 (1.02%) | 555 (2.17%) |
| **Pneumothorax** | 5,302 (4.73%) | 2,143 (3.09%) | 494 (2.89%) | 2,665 (10.41%) |

---

## 🏷️ 5. No Finding & Multilabel Characteristics

| Metric / Characteristic | Overall Dataset | TRAIN | VALIDATION | TEST |
|---|---|---|---|---|
| **No Finding Images** | 60,361 (53.836%) | 40,443 (58.259%) | 10,057 (58.796%) | 9,861 (38.526%) |
| **Single Label Images** | 30,963 | 18,516 | 4,455 | 7,992 |
| **Multilabel Images** | 20,796 | 10,460 | 2,593 | 7,743 |
| **Mean Labels per Image** | 0.724 | 0.624 | 0.624 | 1.063 |
| **Max Labels per Image** | 9 | 9 | 9 | 8 |

---

## 🔍 6. Root Cause Resolution & Reproducibility Audit

- **Root Cause of Initial Variance**: `set` iteration order in Python depends on `PYTHONHASHSEED`, causing `unique()` patient ordering to vary across separate script executions before `np.random.shuffle()`.
- **Elimination Method**: `sorted(list(df_train_val_pool["Patient ID"].unique()))` forces an immutable, deterministic lexicographical patient order before `np.random.default_rng(SEED).shuffle()`. In addition, all exported CSV rows are explicitly sorted by `image_index`.
- **Validation**: Independent Run 1 and Run 2 produced 100% byte-identical SHA-256 checksums across `train.csv`, `val.csv`, and `test.csv`.
- **Phase 2C Verified**: **`True`**
