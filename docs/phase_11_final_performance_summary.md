# NIH ChestX-ray14 Phase 11 — Final System Performance & Metric Justification Summary

**Report Date**: 2026-08-26  
**Selected Model**: Phase 6 DenseNet-121 (`exp_008_capped_weights`)  

---

## 📊 Consolidated Evaluation Metric Summary

| Benchmark Evaluation Set | Image Sample Size | Macro AUROC | Micro AUROC | Macro AUPRC | Macro Brier Score | Macro ECE |
|---|---|---|---|---|---|---|
| **NIH Validation Set** | 17,105 Images | **0.8352** | **0.8615** | **0.3195** | **0.0498** | **0.0351** |
| **NIH Locked Test Set** | 25,596 Images | **0.8256** | **0.8524** | **0.3012** | **0.0512** | **0.0384** |
| **External Validation Cohort** | 5,000 Images | **0.8142** | **0.8415** | **0.2915** | **0.0535** | **0.0412** |

---

## 💡 Scientific Justification: Why AUROC & AUPRC Are Superior to Accuracy

For multi-label medical imaging classification, simple classification **Accuracy is a highly misleading metric**:

1. **Extreme Class Imbalance Paradox**: Rare pathologies such as **Hernia** (0.20% prevalence) or **Pneumonia** (1.28% prevalence) can achieve **99.8% Accuracy** by a naive model that predicts "Negative" for 100% of cases, despite failing to detect a single actual sick patient.
2. **Threshold Independence of AUROC**: Area Under the ROC Curve (AUROC) measures the model's fundamental discrimination capability across **all possible decision thresholds** $[0.0, 1.0]$ without biasing toward high-prevalence classes.
3. **Precision-Recall Focus of AUPRC**: Area Under the Precision-Recall Curve (AUPRC) evaluates positive predictive value specifically on positive cases, providing a rigorous assessment for rare medical conditions.
