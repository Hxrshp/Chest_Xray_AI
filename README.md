# AI-Powered Chest X-Ray Diagnosis & Explainable Medical Report Generation System

A modular, production-ready Deep Learning framework for automated multi-label pathology detection, visual explainability (Grad-CAM), and structured medical report generation from Chest X-Ray images.

---

## 💡 System Architecture Overview (How the Project Will Eventually Work)

The project will operate via an end-to-end automated clinical workflow across 5 key architectural modules:

```text
[Chest X-Ray Input (DICOM/PNG)]
               │
               ▼
   1. Preprocessing Pipeline (Normalisation, Contrast Enhancement, Resizing)
               │
               ▼
   2. Multi-Label Deep Learning Classification (PyTorch Convolutional Backbones)
               │
               ├── Pathology Confidence Scores (Pneumonia, Atelectasis, Cardiomegaly, etc.)
               │
               ▼
   3. Explainable AI Engine (Grad-CAM Heatmap Generation)
               │
               ├── Visualizing Clinical Regions of Interest
               │
               ▼
   4. Medical Report Generator (Structured Clinical Impressions & Findings)
               │
               ▼
   5. FastAPI Backend Service & React + Vite + TypeScript Frontend UI
```

---

## ⚠️ Important Medical AI Governance Rule

> **CRITICAL MEDICAL AI COMPLIANCE**:
> Never fabricate dataset statistics, model metrics, accuracy, AUROC, sensitivity, specificity, predictions, or clinical findings.
> 
> Until validation experiments are performed on clinical datasets in upcoming phases, all quantitative metrics are marked as:
> **`NOT YET MEASURED`**.

---

## 📁 Repository Directory Hierarchy

```text
Chest-Xray-AI/
├── backend/                  # FastAPI Application, Routers, Database Models
├── frontend/                 # React + Vite + TypeScript User Interface
├── ml/                       # Machine Learning Core Pipeline
│   ├── data/                 # Data loading and streaming logic
│   ├── datasets/             # PyTorch Dataset wrappers (NIH / CheXpert)
│   ├── preprocessing/        # Image transforms, CLAHE contrast, normalisation
│   ├── models/               # Neural network models (ResNet, DenseNet, EfficientNet)
│   ├── training/             # Loss functions, optimisers, training loops
│   ├── evaluation/           # ROC-AUC, Sensitivity, Specificity calculators
│   ├── explainability/       # Grad-CAM heatmap visualization engines
│   ├── inference/            # Model loading & report generation pipeline
│   └── utils/                # Config, Logger, Seed, and Reproducibility tools
├── data/                     # Raw DICOM/PNG data storage (Git Ignored)
├── models/                   # Saved model weights & checkpoints (Git Ignored)
├── reports/                  # Generated diagnostic reports & output artifacts
├── tests/                    # PyTest Unit & Integration test suite
├── docs/                     # Documentation & Architecture diagrams
├── scripts/                  # Data download and utility automation scripts
├── docker/                   # Docker Compose services (PostgreSQL)
├── .env.example              # Environment variables template
├── .gitignore                # Git exclusion rules
├── README.md                 # Project Overview & Guide
└── requirements.txt          # Python dependencies
```

---

## 🚀 Virtual Environment Setup Instructions (Python 3.11)

Follow these step-by-step instructions to create and activate your Python virtual environment on Windows:

### Step 1: Check Python Version
Ensure Python 3.11 is installed on your system:
```powershell
python --version
```

### Step 2: Create Virtual Environment
Navigate to `Chest-Xray-AI` root directory and create the `.venv` environment:
```powershell
python -m venv .venv
```

### Step 3: Activate Virtual Environment
On Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If script execution policy blocks activation, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first)*

On Windows Command Prompt (cmd):
```cmd
.venv\Scripts\activate.bat
```

### Step 4: Upgrade Pip & Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔧 Git Repository Initialization Instructions

To initialize Git tracking for this project:

```powershell
# 1. Navigate to the project directory
cd Chest-Xray-AI

# 2. Initialize Git repository
git init

# 3. Check git status to ensure .gitignore is properly masking unwanted files
git status

# 4. Add initial foundation files to staging
git add .

# 5. Make initial commit
git commit -m "feat: initialize Chest-Xray-AI project foundation (Phase 1)"
```

---

## 🧪 How to Verify the Setup

You can run the built-in foundation verification script using Python directly or `pytest`:

### Option A: Run verification script directly
```powershell
python tests/test_foundation.py
```

### Option B: Run via PyTest
```powershell
pytest tests/test_foundation.py
```

### Expected Verification Output:
```text
Running foundation verification checks...
All foundation verification tests PASSED!
```

---

## 📊 Current Project Metrics Status

| Metric | Status |
| :--- | :--- |
| **Model AUROC** | `NOT YET MEASURED` |
| **Sensitivity** | `NOT YET MEASURED` |
| **Specificity** | `NOT YET MEASURED` |
| **F1 Score** | `NOT YET MEASURED` |
| **Dataset Size** | `NOT YET MEASURED` |
