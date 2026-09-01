# NIH ChestX-ray14 Phase 11 — Complete Project Inventory & Asset Index

**Audit Date**: 2026-08-26  
**Status**: **PHASE 11 VERIFIED — COMPLETE INVENTORY AUDITED**  
**Selected Checkpoint**: `checkpoints/phase6/final/best.pth`  

---

## 📋 Comprehensive Canonical Asset Directory

| Asset Category | Canonical Path / File | Purpose & Function | Verification Status |
|---|---|---|---|
| **Raw Dataset** | `data/raw/images/` | 112,120 PNG frontal chest X-rays | VERIFIED |
| **Split Manifests** | `data/processed/manifests/` | Patient-disjoint split files (`train.csv`, `val.csv`, `test.csv`) | VERIFIED |
| **Model Checkpoint** | `checkpoints/phase6/final/best.pth` | Selected DenseNet-121 baseline (`exp_008_capped_weights`) | VERIFIED |
| **Preprocessing Engine** | `ml/inference/preprocessing.py` | ImageNet RGB standardization ($320 \times 320$) | VERIFIED |
| **Predictor Engine** | `ml/inference/predictor.py` | Production 14-class inference & validation thresholding | VERIFIED |
| **Grad-CAM Explainer** | `ml/inference/explainability.py` | Activation heatmaps targeting `denseblock4.denselayer16.conv2` | VERIFIED |
| **Web Application** | `app/main.py` | Streamlit research user interface | VERIFIED |
| **FastAPI Backend** | `app/api.py` | Programmatic `/predict` and `/explain` REST endpoints | VERIFIED |
| **CLI Predictor** | `scripts/predict_image.py` | Single-image prediction CLI tool | VERIFIED |
| **CLI Batch Processor** | `scripts/predict_batch.py` | Batch CSV manifest processor | VERIFIED |
| **CLI Explainer** | `scripts/generate_explanation.py` | Grad-CAM heatmap CLI tool | VERIFIED |
| **Validation Thresholds** | `data/processed/phase_5_validation_thresholds.json` | 14 pathology thresholds derived from Youden's J statistic | VERIFIED |
| **Phase 1-11 Verification Suites** | `scripts/verify_phase_*.py` | Automated verification scripts for all 11 phases | VERIFIED |
| **System Model Card** | `docs/MODEL_CARD.md` | Official AI transparency Model Card | VERIFIED |
| **Final Report** | `docs/FINAL_PROJECT_REPORT.md` | Publication-ready consolidated research report | VERIFIED |
| **Presentation Notes** | `docs/FINAL_PROJECT_PRESENTATION_NOTES.md` | Presentation slide notes for reviewers | VERIFIED |
| **Live Demo Guide** | `docs/PHASE_11_DEMO_GUIDE.md` | Step-by-step live presentation demonstration procedure | VERIFIED |

---

## 🔒 Immutability Assurance Statement

All production assets, checkpoints, threshold files, and code modules listed above are canonical, verified, and locked against modification.
