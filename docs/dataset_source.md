# NIH ChestX-ray14 Dataset Source & Specification

**Document Version**: 1.0  
**Status**: Authoritative Reference  
**Project**: AI-Powered Chest X-Ray Diagnosis & Explainable Medical Report Generation System  

---

## 1. Official Dataset Name
**NIH ChestX-ray14** (expanded version of ChestX-ray8).

---

## 2. Official Source
Hosted and published by the **National Institutes of Health (NIH) Clinical Center**.  
- **Official Download Portal**: [NIH Clinical Center Box Repository](https://nihcc.app.box.com/v/ChestXray-NIHCC)  
- **Official NIH Release Portal**: [NIH Clinical Center Open Data Access](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community)

---

## 3. Dataset Citation

```bibtex
@inproceedings{Wang_2017_CVPR,
  author    = {Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald M.},
  title     = {ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2017},
  pages     = {2097--2106},
  doi       = {10.1109/CVPR.2017.369},
  url       = {https://arxiv.org/abs/1705.02315}
}
```

---

## 4. Dataset Size as Documented by Source
- **Total Images**: 112,120 frontal-view Chest X-ray images (8-bit grayscale PNGs at $1024 \times 1024$ resolution).
- **Total Unique Patients**: 30,805 anonymized patients.
- **Compressed Archive Size**: ~42.1 GB to 45.0 GB split across 12 archives (`images_001.tar.gz` through `images_012.tar.gz`).
- **Uncompressed Size**: ~45.0 GB to 50.0 GB on disk.

---

## 5. Available Metadata
The official dataset metadata is supplied in two primary CSV files:

### A. `Data_Entry_2017.csv` (112,120 rows)
Primary metadata table containing image-level labels and patient demographics.
- `Image Index`: Filename of the X-ray image (e.g. `00000001_000.png`).
- `Finding Labels`: Pipe-delimited list of disease findings (e.g. `Cardiomegaly|Emphysema` or `No Finding`).
- `Follow-up #`: Patient follow-up study number.
- `Patient ID`: Anonymized integer patient identifier.
- `Patient Age`: Patient age in years at the time of study.
- `Patient Gender`: Patient gender (`M` or `F`).
- `View Position`: Radiographic projection view (`PA` - Posteroanterior or `AP` - Anteroposterior).
- `OriginalImage[Width, Height]`: Dimensions of original DICOM image.
- `OriginalImagePixelSpacing[x, y]`: Physical pixel spacing (mm per pixel).

### B. `BBox_List_2017.csv` (984 rows)
Ground-truth bounding box annotations for disease localization benchmarking:
- `Image Index`, `Finding Label`, `BBox_x`, `BBox_y`, `BBox_w`, `BBox_h`.

### C. Split Files
- `train_val_list.txt` (86,524 image filenames)
- `test_list.txt` (25,596 image filenames)

---

## 6. Patient Identifiers
- Unique 8-digit integer format (e.g., `Patient ID: 00000001`).
- Used to enforce strict patient-level splitting across train, validation, and test sets.

---

## 7. Image Identifiers
- String format: `<PATIENT_ID>_<FOLLOWUP_INDEX>.png` (e.g., `00000001_000.png`, `00000001_001.png`).
- **CRITICAL**: The filename contains only the patient ID and study index—it does **NOT** contain pathology labels.

---

## 8. Disease Labels
The dataset contains 14 thoracic pathology categories + 1 "No Finding" category:

| Target Status | Disease Label | Clinical Description |
|---|---|---|
| **Project Target 1** | **Pneumonia** | Inflammatory lung infection filling alveoli with fluid/pus |
| **Project Target 2** | **Cardiomegaly** | Enlarged heart silhouette |
| **Project Target 3** | **Edema** | Accumulation of fluid in lung tissue / alveoli |
| **Project Target 4** | **Pneumothorax** | Air leakage into pleural space causing lung collapse |
| **Project Target 5** | **Atelectasis** | Partial or complete collapse of lung lobe/tissue |
| *Secondary* | Consolidation | Solidification of lung tissue due to fluid/cellular exudate |
| *Secondary* | Infiltration | Abnormal accumulation of substance (fluid, cells) in tissue |
| *Secondary* | Effusion | Fluid accumulation in pleural cavity |
| *Secondary* | Mass | Lesion $>3 \text{ cm}$ in diameter |
| *Secondary* | Nodule | Lesion $\le 3 \text{ cm}$ in diameter |
| *Secondary* | Emphysema | Damage and enlargement of alveoli air sacs |
| *Secondary* | Fibrosis | Scarring of lung tissue |
| *Secondary* | Pleural Thickening| Thickening of pleural membranes |
| *Secondary* | Hernia | Protrusion of organ through surrounding tissue |
| *Baseline* | No Finding | No pathology detected among the 14 categories |

---

## 9. Uncertain Labels
Unlike CheXpert, NIH ChestX-ray14 does not include explicit "Uncertain" (`-1`) labels in `Data_Entry_2017.csv`. Each of the 14 disease categories is evaluated as present or absent based on NLP mining of radiology reports.

---

## 10. Dataset Limitations
1. **Single Projection**: Images are frontal X-rays (PA/AP views) without lateral views.
2. **2D Image Resolution**: Original DICOM images downsampled to 8-bit $1024 \times 1024$ PNGs.
3. **No Clinical Context**: Lacks longitudinal clinical history, vital signs, or laboratory results.

---

## 11. Known Label-Noise Considerations
- **Automatic Extraction**: Labels were generated automatically using NLP (MetaMap and DNorm) from unstructured radiology reports.
- **Estimated Noise Level**: Authors estimate NLP label extraction accuracy at **>90%**, implying up to **~10% label noise** (false positives and false negatives).
- **Clinical Implication**: Models trained on ChestX-ray14 must account for potential label noise during training and evaluation.

---

## 12. Licensing & Terms of Use
- **License**: Public domain / Open academic research access provided by NIH Clinical Center.
- **Terms of Use**:
  1. Users must acknowledge the NIH Clinical Center as the data provider.
  2. Users MUST NOT attempt to re-identify patients.
  3. Non-commercial scientific research and education usage.

---

## 13. Recommended Acquisition Method
Scripted download via official NIH Box repository links or via official NIH script using chunked HTTP retrieval to ensure download stability over high-latency connections.

---

## 14. Expected Files
```text
Data_Entry_2017.csv        (~8.2 MB)
BBox_List_2017.csv         (~45 KB)
train_val_list.txt         (~780 KB)
test_list.txt              (~230 KB)
images_001.tar.gz          (~2.0 GB)
images_002.tar.gz          (~2.0 GB)
images_003.tar.gz          (~2.0 GB)
...
images_012.tar.gz          (~2.0 GB)
```

---

## 15. Expected Directory Structure

```text
Chest-Xray-AI/
└── data/
    ├── raw/
    │   ├── Data_Entry_2017.csv
    │   ├── BBox_List_2017.csv
    │   ├── train_val_list.txt
    │   ├── test_list.txt
    │   └── images/
    │       ├── 00000001_000.png
    │       ├── 00000001_001.png
    │       └── ... (112,120 PNG images)
    └── processed/
```
