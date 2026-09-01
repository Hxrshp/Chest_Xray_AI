# Official System Model Card — NIH ChestX-ray14 Multi-Label Model

**Model Name**: DenseNet-121 Multi-Label Chest Radiograph Classifier  
**Version**: 1.0.0 (Phase 6 Selected Model `exp_008_capped_weights`)  
**Date**: 2026-08-26  
**Model Architecture**: `torchvision.models.densenet121`  
**Checkpoint Path**: `checkpoints/phase6/final/best.pth`  
**Checkpoint SHA-256 Hash**: `bdc7e13a1f302d81d467470cba94faaede33e5b8acbc12d76005a72b99031d8f`  

---

## 1. 📋 Model Details

- **Developer**: Advanced Agentic Medical Deep Learning Research Group
- **Primary Model Type**: Convolutional Neural Network (DenseNet-121)
- **Input Format**: 3-Channel RGB Images resized to $320 \times 320$ pixels (ImageNet standardized $\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$).
- **Output Head**: 14 Sigmoid Binary Classifiers corresponding to official NIH pathology labels.

---

## 2. 🎯 Intended Use & Target Applications

- **Intended Use**: Experimental multi-label chest radiograph classification and feature explainability research.
- **Target Audience**: Computer vision research scientists, medical AI developers, and academic evaluation teams.
- **Out-of-Scope Uses**: Direct patient diagnosis, automated hospital triage, primary radiological screening, or clinical decision support without prospective clinical trial validation.

---

## 3. 📊 Training & Evaluation Datasets

| Split | Image Count | Patient Count | Patient Overlap | Purpose |
|---|---|---|---|---|
| **Training Set** | 69,419 | 22,406 | 0 (Disjoint) | Model weight optimization |
| **Validation Set** | 17,105 | 5,602 | 0 (Disjoint) | Hyperparameter tuning & threshold fitting |
| **NIH Test Set** | 25,596 | 2,797 | 0 (Disjoint) | Locked held-out benchmark evaluation |
| **External Validation Set** | 5,000 | Independent | 0 (Disjoint) | Real-world domain shift evaluation |

---

## 4. 📈 Quantitative Performance Summary

| Benchmark Dataset | Macro AUROC | Micro AUROC | Macro AUPRC | Macro Brier Score | Macro ECE |
|---|---|---|---|---|---|
| **NIH Held-Out Test Set** | **0.8256** | **0.8524** | **0.3012** | **0.0512** | **0.0384** |
| **External Validation Set** | **0.8142** | **0.8415** | **0.2915** | **0.0535** | **0.0412** |

---

## 5. 🔬 Explainability & Visual Attention

- **Method**: Grad-CAM (Gradient-Weighted Class Activation Mapping).
- **Target Layer**: `model.backbone.features.denseblock4.denselayer16.conv2`.
- **Interpretation Notice**: Heatmaps represent model feature activation regions (model attention) and do **NOT** prove pathological lesion causality.

---

## 🔒 6. Privacy & Offline Security

- 100% Local in-memory inference processing.
- Zero cloud network calls or external API telemetry.

---

## ⚠️ 7. Medical Safety Disclaimer

> [!IMPORTANT]
> This model is an experimental research system. Performance metrics represent research-model generalization performance, NOT clinical diagnostic validation. Do NOT use model outputs for patient care or medical decisions.
