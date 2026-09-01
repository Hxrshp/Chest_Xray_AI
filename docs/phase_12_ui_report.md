# NIH ChestX-ray14 Phase 12 — Radiologist Web Interface Architecture Report

**Report Date**: 2026-08-26  
**Status**: **VERIFIED — RADIOLOGIST DECISION-SUPPORT SYSTEM READY**  
**Interface Entry Point**: `app/main.py`  

---

## 🎨 System UI Architecture & Design Principles

The Phase 12 web interface transforms the existing Streamlit research prototype into a clinical decision-support review dashboard built specifically for radiologists and medical imaging researchers.

### Key Architectural Layers
1. **Presentation Layer (`app/main.py`, `app/ui/components.py`, `app/ui/styles.py`)**:
   - Clean, professional medical CSS palette (slate, navy, dark gray).
   - Prominent **RESEARCH USE ONLY — NOT FOR CLINICAL DIAGNOSIS** header badge.
   - Dual-column responsive layout (Image preview on left, findings dashboard on right).
2. **Service Integration Layer (`app/services/`)**:
   - `inference_service.py`: Cached singleton `Predictor` wrapper.
   - `explanation_service.py`: Grad-CAM feature activation overlay generator.
   - `export_service.py`: Machine-readable JSON export builder.
3. **Core ML Layer (`ml/inference/`)**:
   - Production DenseNet-121 model (`checkpoints/phase6/final/best.pth`).
   - ImageNet RGB standardization ($320 \times 320$).
   - Youden's J validation thresholds (`data/processed/phase_5_validation_thresholds.json`).
