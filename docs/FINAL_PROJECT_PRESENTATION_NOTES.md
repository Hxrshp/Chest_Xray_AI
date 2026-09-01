# NIH ChestX-ray14 — Final Project Presentation Notes

**For Project Reviewers, Evaluators, and Oral Defense Panels**  

---

### Slide 1: Problem Statement
Automated multi-label chest radiograph classification poses major computer vision challenges due to severe pathology co-occurrence, class imbalance, diffuse opacity patterns, and label noise.

### Slide 2: Motivation
Chest radiographs are the most common diagnostic imaging modality worldwide. Decision-support research tools can assist in radiograph prioritization and feature visual explainability.

### Slide 3: NIH ChestX-ray14 Dataset
Contains 112,120 frontal chest X-rays from 30,805 unique patients labeled across 14 official pathology classes via NLP text mining from radiological reports.

### Slide 4: Patient-Disjoint Data Split Strategy
To eliminate patient leakage, all radiographs from a single patient were assigned exclusively to one split:
- **Train**: 69,419 images (22,406 patients)
- **Validation**: 17,105 images (5,602 patients)
- **Locked Test**: 25,596 images (2,797 patients)

### Slide 5: Preprocessing & Standardization
Images are standardized to 3-channel RGB, resized to $320 \times 320$ pixels, and normalized using ImageNet channel mean $\mu = [0.485, 0.456, 0.406]$ and std $\sigma = [0.229, 0.224, 0.225]$.

### Slide 6: Model Architecture (DenseNet-121)
DenseNet-121 utilizes dense feature reuse and direct feature maps connections, making it ideal for multiscale anatomical feature extraction (6,968,206 total parameters).

### Slide 7: Multi-Label Classification Head
The network outputs 14 unnormalized logits passed through independent Sigmoid activations to produce probability estimates $p_i \in [0.0, 1.0]$ for each pathology.

### Slide 8: Class Imbalance Capping Strategy
Unweighted loss neglects rare diseases (Hernia: 0.20%), while extreme positive weights ($630.08$) cause gradient instability. Capping positive weights at $\le 50.0$ provided optimal optimization stability.

### Slide 9: Optimizer & Training Strategy
Trained using AdamW optimizer ($\text{lr} = 1\times 10^{-4}$, $\text{weight\_decay} = 1\times 10^{-2}$) and ReduceLROnPlateau scheduler under mild medical data augmentations.

### Slide 10: Validation-Based Model Selection
Model selection was conducted strictly using **Validation Set Macro AUROC** (`0.8352`), keeping the held-out test set 100% locked.

### Slide 11: Held-Out Test Evaluation Results
- **Test Macro AUROC**: **0.8256**
- **Test Micro AUROC**: **0.8524**
- **Test Macro AUPRC**: **0.3012**
- **95% Bootstrap CI**: **[0.8211, 0.8299]**

### Slide 12: External Multi-Center Generalization
Evaluated on an independent 5,000-image external validation cohort:
- **External Macro AUROC**: **0.8142** (Minor ~1.14% domain shift decay across hospital scanners).

### Slide 13: Grad-CAM Visual Feature Explainability
Gradient-weighted class activation mapping targeting `denseblock4.denselayer16.conv2` generates colorized jet heatmaps overlayed on radiographs to visualize model attention regions.

### Slide 14: Streamlit Research Application
A responsive web application (`app/main.py`) provides single-image analysis, probability dashboards, Youden's J thresholding, Grad-CAM viewer, and JSON exports.

### Slide 15: Local Privacy & Security Controls
Inference processes images 100% locally in volatile memory with zero cloud network telemetry.

### Slide 16: System Limitations
Weakest performance observed on diffuse opacities (Infiltration AUROC: 0.6982) and small focal lesions (Nodule AUROC: 0.7321) due to NLP report mining noise.

### Slide 17: Scientific Metric Justification
AUROC and AUPRC evaluate model discrimination across all decision thresholds, avoiding the false 99.8% accuracy illusion caused by class imbalance.

### Slide 18: Medical Safety Statement
Prominently labeled **RESEARCH USE ONLY — NOT FOR CLINICAL DIAGNOSIS**. Real-world medical deployment would require prospective clinical trials and regulatory approval.
