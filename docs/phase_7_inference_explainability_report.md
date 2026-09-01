# NIH ChestX-ray14 Phase 7 — Production Inference & Visual Explainability Technical Report

**Report Date**: 2026-08-26  
**Status**: **PHASE 7 VERIFIED — INFERENCE & EXPLAINABILITY READY**  
**Selected Phase 6 Model**: `exp_008_capped_weights` (DenseNet-121, Capped Weights $\le 50.0$, LR=1e-4)  
**Model Checkpoint**: `checkpoints/phase6/final/best.pth`  

---

## 1. 🏗️ Model Architecture & Inference Preprocessing Contract

- **Architecture**: PyTorch `torchvision.models.densenet121` with 14 output linear head classifiers (`num_classes=14`).
- **Target Resolution**: $320 \times 320$ pixels with Bilinear interpolation.
- **Normalization**: ImageNet standardization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **Supported Image Inputs**: Single-channel Grayscale (`L`), 3-channel (`RGB`), and 4-channel (`RGBA`) DICOM/PNG/JPEG images.
- **Validation Threshold Source**: `data/processed/phase_5_validation_thresholds.json` (Derived exclusively from Youden's J statistic on the 17,105 validation set).

---

## 2. 🔍 Grad-CAM Visual Explainability Pipeline

- **Target Layer**: `model.backbone.features.denseblock4.denselayer16.conv2` (Final $3 \times 3$ convolutional layer in denseblock4 before global average pooling).
- **Activation Map Generation**:
  1. Forward pass computes class logits.
  2. Target class logit backpropagates gradients to `denseblock4`.
  3. Spatial feature maps are weighted by channel-wise mean gradients.
  4. ReLU activation filters negative influences; bilinear interpolation resizes heatmap to original image dimensions ($1024 \times 1024$).
  5. Jet color map overlay is blended at 45% alpha transparency with original radiograph.

---

## 3. 🧪 Robustness & Safety Test Results

| Test Case | Inputs Tested | Behavior / Result | Status |
|---|---|---|---|
| Grayscale Images | Grayscale 8-bit PNGs | Converted safely to 3-channel RGB | PASSED |
| RGBA Images | 4-Channel Transparency PNGs | Alpha channel discarded, RGB preserved | PASSED |
| Resolution Extremes | $64 \times 64$ to $2048 \times 2048$ | Bilinear scaling handles all dimensions | PASSED |
| Corrupt Image Files | Invalid header / truncated streams | Caught gracefully without process crash | PASSED |
| Empty Files | 0-Byte empty PNGs | Rejected with `ValueError` | PASSED |
| Missing Files | Non-existent file paths | Caught with `FileNotFoundError` | PASSED |
| Parameter Immutability | Model weights hash before/after | Model parameters 100% frozen | PASSED |
| Determinism Parity | Single vs Batch probabilities | Probability delta $= 0.0$ (`np.allclose`) | PASSED |

---

## 4. 🔒 Zero-Leakage Audit Summary

- **Train Set**: 69,419 images used for Phase 6 parameter training.
- **Validation Set**: 17,105 images used for Phase 6 model selection and Phase 5 threshold fitting.
- **Test Set**: 25,596 images **LOCKED** (0 test labels used during Phase 7 inference engine development).

---

## ⚠️ Medical Safety & Research Disclaimer

> [!IMPORTANT]
> This system is an experimental multi-label research baseline for chest radiograph analysis. It is **NOT** a clinically validated diagnostic device, certified medical software, or a replacement for a qualified radiologist. Predictions and visual heatmaps are statistical model outputs and must never be used for primary patient diagnosis, automated triage, or direct clinical decision-making.
