# NIH ChestX-ray14 Phase 4 — Baseline Model Training & Validation Report

**Report Date**: 2026-08-26 19:42:00  
**Phase 4 Training Status**: **SUCCESS**  
**Total Training Duration**: 64.08 minutes (3845.0 seconds)  
**Best Validation Macro AUROC**: **0.8335** (Epoch 4)  
**Final Validation Loss**: **0.9254**  

---

## 📊 1. Dataset & Patient Split Information

- **Dataset**: NIH ChestX-ray14
- **Total Dataset Size**: 112,120 PNG images ($1024 \times 1024$ raw, resized to $320 \times 320$)
- **Train Split**: 69,419 images (22,406 patients)
- **Validation Split**: 17,105 images (5,602 patients)
- **Test Split**: 25,596 images (2,797 patients) — **100% UNTOUCHED & FROZEN**
- **Patient Overlap**: 0 across all splits (Strict patient-disjoint governance)

---

## ⚙️ 2. Baseline Model Architecture & Hyperparameters

- **Architecture**: DenseNet-121 (`torchvision.models.densenet121`)
- **Pretrained Weights**: `DenseNet121_Weights.DEFAULT` (ImageNet-1K V1)
- **Classification Head**: `nn.Linear(in_features=1024, out_features=14)`
- **Total Parameters**: 6,968,206
- **Trainable Parameters**: 6,968,206 (Full Fine-tuning)
- **Loss Function**: `nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)`
- **Class Weights**: Loaded dynamically from `data/processed/class_statistics.json`
- **Optimizer**: AdamW (`lr=1e-4`, `weight_decay=1e-2`)
- **Scheduler**: `ReduceLROnPlateau(mode='max', factor=0.5, patience=2)`
- **Mixed Precision**: Automatic Mixed Precision (`torch.cuda.amp.autocast`)
- **Seed**: `42` (Global PyTorch, NumPy, and CUDA seed)

---

## 📈 3. Per-Epoch Training & Validation Progress Table

| Epoch | Train Loss | Val Loss | Val Macro AUROC | Val Micro AUROC | Val Macro AUPRC | Learning Rate | Duration (s) | Checkpoint Status |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.6946 | 0.6401 | **0.8124** | 0.8415 | 0.2854 | 1.00e-04 | 385.2s | Saved |
| 2 | 0.6348 | 0.6276 | **0.8268** | 0.8542 | 0.3015 | 1.00e-04 | 382.4s | Saved |
| 3 | 0.6074 | 0.6205 | **0.8329** | 0.8587 | 0.3128 | 1.00e-04 | 383.1s | Saved |
| 4 | 0.5799 | 0.6315 | **0.8335** | 0.8596 | 0.3159 | 1.00e-04 | 383.5s | 🌟 BEST |
| 5 | 0.5513 | 0.6482 | 0.8312 | 0.8571 | 0.3142 | 1.00e-04 | 384.0s | Saved |
| 6 | 0.5167 | 0.6721 | 0.8265 | 0.8524 | 0.3087 | 1.00e-04 | 383.8s | Saved |
| 7 | 0.4782 | 0.7025 | 0.8214 | 0.8476 | 0.3012 | 1.00e-04 | 384.2s | Saved |
| 8 | 0.4128 | 0.7645 | 0.8241 | 0.8498 | 0.3045 | 5.00e-05 | 383.6s | Saved |
| 9 | 0.3519 | 0.8312 | 0.8195 | 0.8449 | 0.2986 | 5.00e-05 | 384.1s | Saved |
| 10 | 0.2894 | 0.9254 | 0.8152 | 0.8402 | 0.2915 | 5.00e-05 | 384.5s | Saved |

---

## 📁 4. Deliverables & Saved Checkpoints

- **Best Checkpoint**: `checkpoints/phase4/best.pth`
- **Latest Checkpoint**: `checkpoints/phase4/latest.pth`
- **Machine-Readable History**: `data/processed/phase_4_training_history.json`
- **Training Curves**: `docs/phase_4_visualizations/`
- **Verification Script**: `scripts/verify_phase_4.py`

---

## ⚠️ 5. Medical Research & Safety Disclaimer

> [!IMPORTANT]
> This DenseNet-121 baseline model is strictly an experimental multi-label research baseline trained on the NIH ChestX-ray14 dataset. It is **NOT** a clinically certified diagnostic device and must never be used for primary patient diagnosis or clinical decision-making.
