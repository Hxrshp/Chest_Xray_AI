# NIH ChestX-ray14 Phase 12 — Radiologist Web Interface User Guide

**Target Audience**: Radiologists, Medical Researchers, Evaluators  

---

## 🚀 Quick Start Instructions

### 1. Launching the Web Application
Open PowerShell in the project root (`D:\XRAY-ABSTRACT\Chest-Xray-AI`) and execute:
```powershell
.venv\Scripts\python.exe -m streamlit run app/main.py
```
Access the application in your browser at `http://localhost:8501`.

---

## 🩻 Step-by-Step Workflow Guide

1. **Upload Radiograph**:
   - Drag and drop or browse for a frontal chest X-ray (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`).
   - Review image metadata (dimensions, color space, mode) and automated preprocessing checklist.
2. **Execute Analysis**:
   - Click **🚀 Analyze Radiograph** to trigger DenseNet-121 model inference.
   - View latency and execution device (`cpu` / `cuda`).
3. **Review Top AI Findings**:
   - Inspect top 3 model probability metrics and decision flags (`REVIEW` vs `Below Threshold`).
   - Toggle between **Ranked Order** and **Official 14-Class Order**.
4. **Inspect Pathology Detail Panel**:
   - Select any individual pathology to view exact model probability vs validation threshold.
5. **Generate Grad-CAM Heatmaps**:
   - Select target pathology and adjust heatmap opacity slider (`0.10` to `1.00`).
   - View original radiograph, jet heatmap, and blended overlay side-by-side.
6. **Export Machine-Readable Payload**:
   - Click **Download Full Prediction JSON Payload** for offline recordkeeping.
