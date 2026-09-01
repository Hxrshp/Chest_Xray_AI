# NIH ChestX-ray14 Phase 12 — Radiologist Decision-Support Workflow Protocol

**Protocol Status**: **RESEARCH DECISION SUPPORT PROTOCOL**  

---

## ⚕️ Clinical Decision-Support Communication Protocol

### 1. Fundamental Principle of Human-in-the-Loop AI
The Chest-Xray-AI system operates strictly as a **second-reader decision-support tool**. The attending radiologist retains complete clinical autonomy and primary diagnostic responsibility.

### 2. Standardized AI Terminology Standard

| Metric / Output | Correct Decision-Support Terminology | Strictly Prohibited Terms |
|---|---|---|
| Model Output Value | `Model Probability` / `AI Score` | `Diagnostic Certainty` / `Confirmed Disease` |
| Binary Decision Flag | `FLAG: REVIEW` / `Threshold-Positive` | `Patient has Disease` / `Confirmed Positive` |
| Below Threshold Output | `Below Threshold` / `Low AI Priority` | `Ruled Out` / `Normal Patient` |
| Grad-CAM Visualization | `Model Feature Attention Overlay` | `Confirmed Pathological Lesion Map` |

---

## 🔒 Privacy & Offline Security Assurance
- 100% Local in-memory execution.
- 0 Cloud telemetry or external image API transmissions.
