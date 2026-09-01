# NIH ChestX-ray14 Dataset Acquisition & Storage Strategy Plan

**Document Version**: 1.0  
**Status**: Strategic Planning  
**Project**: AI-Powered Chest X-Ray Diagnosis & Explainable Medical Report Generation System  

---

## A. Source
- **Provider**: National Institutes of Health (NIH) Clinical Center.
- **Repository URL**: `https://nihcc.app.box.com/v/ChestXray-NIHCC`
- **Primary Citation**: Wang et al., CVPR 2017 (`arXiv:1705.02315`).

---

## B. Acquisition Method
1. **Metadata Acquisition**: Download `Data_Entry_2017.csv`, `BBox_List_2017.csv`, `train_val_list.txt`, and `test_list.txt` directly to `data/raw/`.
2. **Image Archive Acquisition**: Scripted batch HTTP download of 12 tarball parts (`images_001.tar.gz` through `images_012.tar.gz`) into `data/raw/archives/` using Python `urllib` / `requests` with chunked streaming and resume capability.

---

## C. Storage Requirements & Machine Feasibility Analysis

### Storage Breakdown
- **Compressed Download Size**: ~42.5 GB (12 tarballs).
- **Uncompressed PNG Size**: ~45.0 GB ($112,120 \text{ images} \times \sim 400 \text{ KB}$ per $1024 \times 1024$ 8-bit PNG).
- **Peak Storage Required (Archive + Uncompressed)**: ~87.5 GB.
- **Optimized Storage Required (Sequential Extract & Delete Archive)**: ~45.0 GB.

### System Feasibility Assessment on Local PC
- **Available Free Space on Drive `D:`**: >100 GB available.
- **Conclusion**: The complete 112,120 image dataset is **manageable** on drive `D:`, provided we use a **sequential extraction pipeline** (extract archive $N$, verify contents, delete archive $N$, repeat for $N=1..12$) to keep maximum disk usage below **48 GB** at any point.

---

## D. Verification Method
1. **Archive Checksum Verification**: Verify downloaded tarball MD5 hashes against published NIH MD5 manifests before extraction.
2. **Row Count Verification**: Verify `Data_Entry_2017.csv` contains exactly **112,120** data rows (112,121 total lines including header).
3. **Corrupt File Scan**: Run a PIL image header check on extracted PNGs to detect incomplete or corrupt files.

---

## E. Extraction Procedure
```powershell
# Conceptual sequential extraction workflow (to be executed in Phase 2B script)
for ($i=1; $i -le 12; $i++) {
    $num = "{0:D3}" -f $i
    $tarFile = "data/raw/archives/images_$num.tar.gz"
    
    # 1. Download tar archive $num
    # 2. Verify MD5 checksum
    # 3. Extract to data/raw/images/
    # 4. Remove $tarFile to free disk space
}
```

---

## F. Metadata Files Required
- `Data_Entry_2017.csv` (Primary index with multi-label annotations)
- `BBox_List_2017.csv` (Ground-truth bounding boxes)
- `train_val_list.txt` (Official train/val split candidate list)
- `test_list.txt` (Official test split candidate list)

---

## G. Expected Image Count
- **Total Images**: **112,120** PNG files.

---

## H. Expected Patient Count
- **Total Unique Patients**: **30,805** patients.

---

## I. Dataset Completeness & Governance Verification

### Completeness Criteria
1. Total files in `data/raw/images/` must equal **112,120**.
2. 100% of filenames listed under `Image Index` in `Data_Entry_2017.csv` must exist in `data/raw/images/`.

### 🛡️ Critical Medical Data Rule
- Filenames (e.g. `00000001_000.png`) encode ONLY `Patient ID` and `Follow-up Index`.
- **Pathology labels MUST be read strictly from `Data_Entry_2017.csv` `Finding Labels` column.** Filenames will never be parsed for disease inferencing.

### 🛡️ Critical Data Leakage Prevention Rule
- **Patient-Level Splitting is Mandatory**: All images belonging to a single `Patient ID` must reside strictly in ONE set (Train, Validation, OR Test).
- No patient will ever overlap across sets.
- Train/Validation/Test split CSV generation will be executed during dataset creation in Phase 2B (no split CSV files created in Phase 2A).
