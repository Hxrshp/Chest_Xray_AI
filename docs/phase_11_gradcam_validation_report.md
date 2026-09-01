# NIH ChestX-ray14 Phase 11 — Grad-CAM Visual Explainability Validation Report

**Audit Date**: 2026-08-26  
**Status**: **VERIFIED — GRAD-CAM EXPLAINABILITY VALIDATED**  
**Module**: `ml/inference/explainability.py`  
**Target Layer**: `model.backbone.features.denseblock4.denselayer16.conv2`  

---

## 🔬 Grad-CAM Validation Checklist

| Test Item | Specification | Result / Finding | Status |
|---|---|---|---|
| Target Convolutional Layer | `denseblock4.denselayer16.conv2` | Final $3 \times 3$ conv layer in DenseNet-121 | PASSED |
| Heatmap Dimension Matching | Resized to input dimensions | Output shape matches original PIL size ($1024 \times 1024$) | PASSED |
| Heatmap Bounded Values | Range $[0.0, 1.0]$ | All values finite and bounded in $[0.0, 1.0]$ | PASSED |
| Colorized Jet Overlay | Image.blend 45% alpha | Smooth color overlay generated | PASSED |
| Model Weight Immutability | Weight hash before/after | Model parameters 100% frozen | PASSED |
| Target Pathology Support | All 14 classes | Supports any requested or highest pathology | PASSED |

---

## 💡 Attention Visualization Disclaimer

> [!IMPORTANT]
> Grad-CAM heatmaps represent model feature activation regions (model attention) and do **NOT** prove the presence, exact boundary, or causal etiology of a pathological lesion.
