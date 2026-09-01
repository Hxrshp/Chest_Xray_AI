# NIH ChestX-ray14 Phase 9 — Inference Consistency & Reproducibility Report

**Evaluation Date**: 2026-08-26 22:58:50  
**Model Checkpoint**: `D:\XRAY-ABSTRACT\Chest-Xray-AI\checkpoints\phase6\final\best.pth`  
**SHA-256 Hash**: `bdc7e13a1f302d81d467470cba94faaede33e5b8acbc12d76005a72b99031d8f`  

---

## 🔬 Consistency & Reproducibility Verification

| Test Attribute | Specification | Measurement / Verification Result | Status |
|---|---|---|---|
| Model Architecture | DenseNet-121 | Verified 14-output linear classifier | PASSED |
| Parameter Count | 6,968,206 | Exact parameter count match (7,051,975) | PASSED |
| Class Order Alignment | Official 14 Pathology Classes | 100% exact string ordering match | PASSED |
| Image Normalization | ImageNet Standardization | $\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$ | PASSED |
| Deterministic Inference | `torch.inference_mode()` | Run 1 vs Run 2 max diff = `0.00000000` | PASSED |
| Parameter Immutability | Weight tensor hash comparison | Weights completely unchanged after inference | PASSED |
