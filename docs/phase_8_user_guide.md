# NIH ChestX-ray14 Application — User Guide & Manual

**Software Version**: 1.0.0 Research Prototype  
**Model Backend**: DenseNet-121 (Phase 6 Selected Baseline)  

---

## 🚀 1. Launching the Application

To launch the web interface, execute the following command from the project root:

```bash
python -m streamlit run app/main.py
```

Alternatively, use the provided launcher script:

```bash
python scripts/run_app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## 🩻 2. Analyzing a Chest Radiograph

1. **Upload Radiograph**: Click **Browse files** or drag-and-drop a PNG, JPG, or JPEG chest X-ray image into the file uploader.
2. **Preview Image**: Inspect original image dimensions, color mode, and target resolution ($320 \times 320$).
3. **Execute Analysis**: Click **🚀 Analyze Radiograph**.
4. **Inspect Results**:
   - **Top Findings**: View the top 3 highest-probability pathology predictions.
   - **All 14 Classes**: Toggle between probability-ranked order and official class order. Inspect probability progress bars and validation-derived threshold predictions (`POSITIVE` vs `Negative`).

---

## 🔬 3. Generating Grad-CAM Visual Heatmaps

1. Scroll down to the **Grad-CAM Visual Feature Explainability** section.
2. Select a target pathology from the dropdown list.
3. Click **Generate Grad-CAM Overlay**.
4. View side-by-side original radiograph and colorized activation heatmap overlay.

---

## 💾 4. Exporting Results

Click **Download Full Prediction JSON** at the bottom of the page to save a machine-readable JSON payload containing timestamp, SHA-256 image hash, model checkpoint metadata, all 14 pathology probabilities, validation thresholds, and medical disclaimers.

---

## ⚠️ Medical Safety & Research Use Notice

> [!IMPORTANT]
> This application is intended strictly for research and evaluation purposes. It is **NOT** a medical device and must **NEVER** be used for clinical diagnosis or patient care.
