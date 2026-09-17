"""
NIH ChestX-ray14 Main Streamlit Application
--------------------------------------------
Radiologist Decision-Support System prototype web application for multi-label chest X-ray classification and visual explainability.
"""

import sys
import time
from pathlib import Path
from PIL import Image
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.ui.styles import CUSTOM_CSS
from app.ui.components import (
    render_header,
    render_uploaded_image_preview,
    render_results_dashboard,
    render_gradcam_section,
    render_pathology_detail_panel,
    render_model_info,
    render_export_section,
)
from app.services.inference_service import run_inference
from app.services.explanation_service import generate_gradcam_explanation


def main():
    st.set_page_config(
        page_title="Chest X-ray AI — Radiologist Decision-Support System",
        page_icon="🩻",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Render Main Header
    render_header()

    # Sidebar Options & Model Info
    with st.sidebar:
        st.header("🩻 Navigation & System Info")
        st.caption("Radiologist Decision-Support Prototype v1.2.0")
        
        st.markdown("""
        **System Governance**:
        - **Model Architecture**: DenseNet-121 (6.9M params)
        - **NIH Test Macro AUROC**: `0.8256`
        - **External Macro AUROC**: `0.8142`
        - **Local Privacy**: 100% Offline / In-Memory
        """)
        
        render_model_info()

        with st.expander("🛡️ Privacy & Security Statement"):
            st.markdown("""
            - All image processing occurs **strictly in local system RAM**.
            - No patient radiographs or metadata are uploaded to cloud servers.
            - No remote API telemetry is active.
            """)

        with st.expander("⚠️ Medical Safety & Limitations"):
            st.markdown("""
            - **RESEARCH USE ONLY** — Not for clinical diagnosis.
            - Model predictions are statistical probability estimates.
            - Final interpretation must be made by a qualified radiologist.
            """)

    # Input Mode Selection
    input_mode = st.radio(
        "Choose Radiograph Input Source:",
        ["📁 Upload Chest Radiograph", "🧪 Select Real Clinical Benchmark Sample"],
        horizontal=True
    )

    pil_img = None
    image_bytes = None
    filename = None

    if input_mode == "📁 Upload Chest Radiograph":
        uploaded_file = st.file_uploader(
            "Upload Chest Radiograph (PNG, JPG, JPEG, BMP, TIFF):",
            type=["png", "jpg", "jpeg", "bmp", "tiff", "tif"],
            help="Select a chest X-ray image file for AI-assisted multi-label analysis."
        )
        if uploaded_file is not None:
            image_bytes = uploaded_file.read()
            uploaded_file.seek(0)
            if len(image_bytes) == 0:
                st.error("Unable to process this image. The uploaded file is empty (0 bytes).")
            else:
                pil_img = Image.open(uploaded_file).convert("RGB")
                filename = uploaded_file.name
    else:
        sample_options = {
            "🫁 Viral Pneumonia (COVID-19 Case)": "real_Pneumonia_Viral_COVID-19_auntminnie-a-2020_01_28_23_51_6665_2020_01_28_Vietnam_coronavirus.jpeg",
            "🫁 Bacterial Pneumonia (Streptococcus)": "real_Pneumonia_Bacterial_Streptococcus_streptococcus-pneumoniae-pneumonia-1.jpg",
            "🫁 Fungal Pneumonia (Pneumocystis)": "real_Pneumonia_Fungal_Pneumocystis_pneumocystis-pneumonia-2-PA.png",
            "🫁 Severe ARDS / Bilateral Opacity": "real_Pneumonia_ARDSSevere.png",
            "🫀 Cardiomegaly (Enlarged Cardiac Silhouette)": "sample_cardiomegaly_openi.png",
            "💧 Pleural Effusion (Fluid Accumulation)": "sample_effusion_openi.png",
            "🩺 Healthy / No Finding Control": "real_No_Finding_F051E018-DAD1-4506-AD43-BE4CA29E960B.jpeg",
            "🔬 NIH ChestX-ray14 Baseline Sample": "sample_nih_00000001_000.png"
        }
        selected_label = st.selectbox("Select Clinical Benchmark Case:", list(sample_options.keys()))
        selected_file = PROJECT_ROOT / "data" / "samples" / sample_options[selected_label]
        if selected_file.exists():
            with open(selected_file, "rb") as f:
                image_bytes = f.read()
            pil_img = Image.open(selected_file).convert("RGB")
            filename = selected_file.name

    if pil_img is not None and image_bytes is not None:
        try:
            # Display Image Preview & Preprocessing Transparency Checklist
            render_uploaded_image_preview(pil_img, filename)

            # Analyze Button
            st.markdown("---")
            if st.button("🚀 Analyze Radiograph", type="primary", use_container_width=True):
                with st.spinner("Executing DenseNet-121 inference model..."):
                    start_t = time.time()
                    result = run_inference(pil_img)
                    elapsed = time.time() - start_t

                    st.session_state["last_result"] = result
                    st.session_state["last_pil"] = pil_img
                    st.session_state["last_bytes"] = image_bytes
                    st.session_state["last_elapsed"] = elapsed
                    st.session_state["last_filename"] = filename

            # Render Results if available in session_state
            if "last_result" in st.session_state and st.session_state.get("last_filename") == filename:
                result = st.session_state["last_result"]
                pil_img = st.session_state["last_pil"]
                image_bytes = st.session_state["last_bytes"]
                elapsed = st.session_state["last_elapsed"]

                render_results_dashboard(result, inference_time_sec=elapsed)
                render_pathology_detail_panel(result)
                render_gradcam_section(pil_img, result)
                render_export_section(result, image_bytes=image_bytes, inference_time_sec=elapsed)

        except Exception as e:
            st.error(f"Unable to process this image: {e}")
    else:
        st.info("👆 Please select or upload a chest X-ray radiograph above to begin AI-assisted review.")


if __name__ == "__main__":
    main()
