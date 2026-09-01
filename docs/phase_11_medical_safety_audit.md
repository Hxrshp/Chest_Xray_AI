# NIH ChestX-ray14 Phase 11 — Responsible AI & Medical Safety Audit Report

**Audit Date**: 2026-08-26  
**Audit Status**: **PASSED — RESPONSIBLE AI SAFETY AUDITED**  

---

## ⚠️ Medical Safety Messaging Checklist

- [x] Prominent `RESEARCH USE ONLY — NOT FOR CLINICAL DIAGNOSIS` disclaimers present on Streamlit UI header, sidebar, dashboard, and export files.
- [x] Probabilities described as statistical model scores (`Model Probability: 0.XX`), NEVER as confirmed medical diagnoses.
- [x] Binary decision outputs formatted as `Model Prediction: Positive / Negative`, strictly avoiding clinical certainty terms (`Confirmed`, `Ruled Out`, `Diagnosed`).
- [x] Grad-CAM overlays explicitly documented as model feature attention regions, NOT causal diagnostic lesion maps.
- [x] Required disclaimer that real-world deployment would demand prospective multi-site clinical trials and regulatory certification.
