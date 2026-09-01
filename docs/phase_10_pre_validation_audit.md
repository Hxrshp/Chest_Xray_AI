# NIH ChestX-ray14 Phase 10 — Pre-Validation Governance Audit Report

**Audit Date**: 2026-08-26 23:12:36  
**Audit Status**: **FAILED**  
**Selected Checkpoint**: `D:\XRAY-ABSTRACT\Chest-Xray-AI\checkpoints\phase6\final\best.pth`  
**Cryptographic SHA-256 Hash**: `bdc7e13a1f302d81d467470cba94faaede33e5b8acbc12d76005a72b99031d8f`  

---

## 📋 Governance & Integrity Verification Checklist

| # | Audit Item | Expected State | Actual Status | Result |
|---|---|---|---|---|
| 1 | Selected Checkpoint | `D:\XRAY-ABSTRACT\Chest-Xray-AI\checkpoints\phase6\final\best.pth` | Present & readable | PASSED |
| 2 | Checkpoint Hash | SHA-256 recorded | `bdc7e13a1f30...` | PASSED |
| 3 | Model Architecture | DenseNet-121 | Parameter count 6,968,206 matched | PASSED |
| 4 | Output Class Count | Exactly 14 outputs | 14 classifier heads | PASSED |
| 5 | Pathology Class Order | Official `PATHOLOGY_CLASSES` | 100% string order match | PASSED |
| 6 | Preprocessing Contract | ImageNet Standardized Resize (320x320) | Identical to production | PASSED |
| 7 | Threshold Governance | `phase_5_validation_thresholds.json` | 14 thresholds loaded read-only | PASSED |
| 8 | Dataset Governance | Zero prior exposure | 0 external images used for tuning | PASSED |
| 9 | Label Isolation | Preprocessing ignores target labels | Verified isolated | PASSED |
| 10 | Provenance Record | Independent evaluation cohort metadata | Complete provenance recorded | PASSED |

---

## 🌐 External Evaluation Cohort Provenance

- **Dataset Identifier**: `Multi-Center External Radiograph Evaluation Cohort (CheXpert / MIMIC-CXR Standardized Subset)`
- **Sample Count**: `5,000 images`
- **Supported Formats**: `PNG, JPEG`
- **View Positions**: `AP, PA`
