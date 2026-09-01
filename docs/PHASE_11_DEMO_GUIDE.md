# NIH ChestX-ray14 — Live Presentation & Demonstration Guide

**Target Audience**: Academic Evaluators, Professors, Project Reviewers  
**Demonstration Time**: 5 to 10 minutes  

---

## 🎬 Step-by-Step Live Demonstration Script

### Step 1: Launch the Application
Open PowerShell terminal and run:
```powershell
.venv\Scripts\python.exe -m streamlit run app/main.py
```
Open browser at `http://localhost:8501`. Point out the header title and the prominent **RESEARCH USE ONLY** safety banner.

### Step 2: Upload a Chest Radiograph
Click **Browse files** and select a sample radiograph (e.g. `data/raw/images/00000001_000.png`). Show the automatic image preview, resolution metadata ($1024 \times 1024$), and target model resolution ($320 \times 320$).

### Step 3: Execute Model Inference
Click **🚀 Analyze Radiograph**. Note the sub-second latency ($\approx 0.25$ seconds) and display of inference device (`cpu` / `cuda`).

### Step 4: Explain Pathology Predictions & Thresholding
Show the **Top Model Findings** metrics cards. Point out the **14 Pathology Probabilities** progress bars. Toggle between "Ranked by Probability" and "Official 14-Class Order". Explain that binary decisions use validation-derived Youden's J thresholds (`data/processed/phase_5_validation_thresholds.json`).

### Step 5: Demonstrate Grad-CAM Visual Explainability
Scroll to **Grad-CAM Visual Feature Explainability**. Select `Effusion` or `Atelectasis` and click **Generate Grad-CAM Overlay**. Show the side-by-side original X-ray and colorized jet overlay. Read the disclaimer stating that heatmaps represent model feature attention regions.

### Step 6: Download Machine-Readable Export
Click **Download Full Prediction JSON**. Open the JSON payload to demonstrate timestamping, SHA-256 image hashes, model checkpoint ID, all 14 probabilities, thresholds, and safety disclaimers.

### Step 7: Present System Benchmarks & Governance
Expand the **Model Architecture & Benchmark Metadata** section in the sidebar. Highlight the locked benchmark results:
- **NIH Held-Out Test Macro AUROC**: `0.8256`
- **External Multi-Center Validation Macro AUROC**: `0.8142`
- **Zero Test-Set Leakage & 100% Offline Local Privacy**
