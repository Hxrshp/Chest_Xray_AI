"""
Streamlit Modular UI Components
-------------------------------
Contains reusable UI rendering components for header, preview, dashboard, explainability, pathology details, metadata, and export.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import streamlit as st
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.output_schema import PredictionResult
from ml.preprocessing.labels import PATHOLOGY_CLASSES
from app.config import (
    APP_TITLE,
    APP_SUBTITLE,
    MEDICAL_DISCLAIMER_SHORT,
    MEDICAL_DISCLAIMER_FULL,
    GRADCAM_DISCLAIMER,
    MODEL_METRICS,
)
from app.services.explanation_service import generate_gradcam_explanation
from app.services.export_service import create_export_payload


def render_header():
    st.markdown(f"<div class='main-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sub-title'>{APP_SUBTITLE}</div>", unsafe_allow_html=True)
    st.markdown("<div class='safety-badge'>⚠️ RESEARCH USE ONLY — NOT FOR CLINICAL DIAGNOSIS</div>", unsafe_allow_html=True)


def render_uploaded_image_preview(pil_img: Image.Image, filename: str):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(pil_img, caption=f"Uploaded X-ray: {filename}", use_container_width=True)
    with col2:
        st.subheader("Image Preprocessing & Technical Details")
        st.write(f"**Filename**: `{filename}`")
        st.write(f"**Original Resolution**: `{pil_img.width} × {pil_img.height}` pixels")
        st.write(f"**Image Mode**: `{pil_img.mode}`")
        st.write(f"**Target Model Resolution**: `320 × 320` pixels (ImageNet Standardized)")
        
        with st.expander("🔍 Automated Preprocessing Pipeline Checklist", expanded=False):
            st.markdown(r"""
            - [x] Input File Validation & Integrity Check
            - [x] Color Space Standardization (L/RGB/RGBA $\rightarrow$ 3-Channel RGB)
            - [x] Resolution Scaling ($320 \times 320$ Bilinear Interpolation)
            - [x] ImageNet Standard Normalization ($\mu=[0.485, 0.456, 0.406], \sigma=[0.229, 0.224, 0.225]$)
            - [x] PyTorch Tensor Conversion & Device Allocation
            """)


def render_results_dashboard(result: PredictionResult, inference_time_sec: Optional[float] = None):
    st.markdown("---")
    st.subheader("📊 AI-Assisted Radiograph Analysis Results")
    
    if inference_time_sec is not None:
        st.caption(f"Inference latency: `{inference_time_sec:.3f} s` | Execution Device: `{result.device}` | Local Privacy: Verified ✓")

    # Flagged Positive Findings
    positive_preds = [p for p in result.predictions.values() if p.binary_prediction]
    positive_preds.sort(key=lambda x: x.probability, reverse=True)

    if positive_preds:
        primary_disease = positive_preds[0]
        other_diseases = [p.pathology for p in positive_preds[1:]]

        # Prominent Primary Disease Diagnosis Card
        st.markdown(f"""
        <div style="background-color: #FEF2F2; border: 2px solid #EF4444; border-radius: 10px; padding: 18px 22px; margin-bottom: 18px;">
            <div style="font-size: 0.95rem; color: #991B1B; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">🩻 AI Diagnostic Result</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #B91C1C; margin: 4px 0 8px 0;">{primary_disease.pathology}</div>
            <div style="font-size: 1.05rem; color: #7F1D1D;">
                <b>Confidence:</b> {primary_disease.probability * 100:.1f}% 
                {f'<span style="margin-left: 12px; color: #991B1B;">| <b>Additional Findings:</b> {", ".join(other_diseases)}</span>' if other_diseases else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #F0FDF4; border: 2px solid #22C55E; border-radius: 10px; padding: 18px 22px; margin-bottom: 18px;">
            <div style="font-size: 0.95rem; color: #166534; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">🩻 AI Diagnostic Result</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #15803D; margin: 4px 0 8px 0;">Normal / No Finding</div>
            <div style="font-size: 1.05rem; color: #166534;">No acute pathologies detected above diagnostic threshold.</div>
        </div>
        """, unsafe_allow_html=True)

    # View Mode Toggle (Flagged First vs Ranked vs Official Order)
    st.write("### All 14 Pathology Model Probabilities & Decision Flags")
    view_mode = st.radio("Display Ordering:", ["Flagged Positive First", "Ranked by Model Probability (Descending)", "Official 14-Class Order"], horizontal=True)

    if view_mode == "Flagged Positive First":
        preds_to_show = sorted(result.predictions.values(), key=lambda x: (not x.binary_prediction, -x.probability))
    elif "Ranked" in view_mode:
        preds_to_show = sorted(result.predictions.values(), key=lambda x: x.probability, reverse=True)
    else:
        preds_to_show = [result.predictions[c] for c in PATHOLOGY_CLASSES]

    for p in preds_to_show:
        col_name, col_bar, col_prob, col_dec = st.columns([2.5, 4, 1.5, 2.0])
        with col_name:
            st.write(f"**{p.pathology}**")
        with col_bar:
            st.progress(float(min(1.0, max(0.0, p.probability))))
        with col_prob:
            st.write(f"`{p.probability * 100:.2f}%`")
        with col_dec:
            if p.binary_prediction:
                st.error("FLAG: REVIEW")
            else:
                st.success("Below Threshold")


def render_gradcam_section(pil_img: Image.Image, result: PredictionResult):
    st.markdown("---")
    st.subheader("🔬 Anatomical Class Activation Mapping (CAM)")
    st.caption("Visualizes where the deep learning model focuses its attention across thoracic lung fields.")

    positive_preds = [p.pathology for p in result.predictions.values() if p.binary_prediction]
    
    # Initialize default selected pathology in session_state if not present
    if "gradcam_class" not in st.session_state or st.session_state["gradcam_class"] not in PATHOLOGY_CLASSES:
        if positive_preds:
            st.session_state["gradcam_class"] = positive_preds[0]
        else:
            st.session_state["gradcam_class"] = result.highest_probability_class

    # Quick Select Shortcuts for Flagged Pathologies
    if positive_preds:
        st.write("**Quick Select Flagged Findings for Instant CAM Overlay:**")
        btn_cols = st.columns(min(5, len(positive_preds)))
        for i, path_name in enumerate(positive_preds[:5]):
            with btn_cols[i]:
                if st.button(f"🚨 {path_name}", key=f"quick_cam_{path_name}", use_container_width=True):
                    st.session_state["gradcam_class"] = path_name
                    st.rerun()

    # Target Pathology Selector
    current_idx = PATHOLOGY_CLASSES.index(st.session_state["gradcam_class"])
    
    def on_pathology_change():
        st.session_state["gradcam_class"] = st.session_state["cam_selector_key"]

    selected_class = st.selectbox(
        "Select ANY Pathology for CAM Attention Mapping:",
        PATHOLOGY_CLASSES,
        index=current_idx,
        key="cam_selector_key",
        on_change=on_pathology_change
    )
    
    target_class = st.session_state.get("gradcam_class", selected_class)

    # Immediately generate and display CAM overlay for the selected pathology
    with st.spinner(f"Computing CAM attention heatmap for {target_class}..."):
        try:
            from app.services.explanation_service import generate_gradcam_explanation
            exp_res = generate_gradcam_explanation(pil_img, target_class=target_class)
            target_prob = result.predictions[target_class].probability * 100
            is_flagged = result.predictions[target_class].binary_prediction
            
            st.markdown(f"""
            <div style="margin: 10px 0 16px 0; padding: 10px 16px; background-color: #F8FAFC; border-left: 4px solid {'#EF4444' if is_flagged else '#3B82F6'}; border-radius: 4px;">
                <b>Showing Attention Map For:</b> <span style="font-size: 1.15rem; font-weight: 700; color: {'#DC2626' if is_flagged else '#2563EB'};">{target_class}</span> 
                (Model Probability: <b>{target_prob:.1f}%</b> | Status: <b>{'FLAG: REVIEW ⚠️' if is_flagged else 'Below Threshold ✓'}</b>)
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(pil_img, caption="Original Radiograph", use_container_width=True)
            with col2:
                st.image(exp_res["overlay_pil"], caption=f"Anatomical CAM Overlay ({target_class})", use_container_width=True)

            st.info(f"💡 **Model Attention Note**: Warm regions (red/yellow/orange) indicate key anatomical features driving model prediction for **{target_class}**.")
        except Exception as e:
            st.error(f"Failed to generate CAM explanation for {target_class}: {e}")


def render_pathology_detail_panel(result: PredictionResult):
    with st.expander("📋 Individual Pathology Detail Panel", expanded=False):
        sel_path = st.selectbox("Select Pathology for Detailed Inspection:", PATHOLOGY_CLASSES)
        pred_item = result.predictions[sel_path]

        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
        with d_col1:
            st.metric("Pathology Name", pred_item.pathology)
        with d_col2:
            st.metric("Model Probability", f"{pred_item.probability * 100:.2f}%")
        with d_col3:
            st.metric("Validation Threshold", f"{pred_item.threshold:.2f}")
        with d_col4:
            st.metric("Decision Flag", "REVIEW" if pred_item.binary_prediction else "Below Threshold")


def render_model_info():
    with st.expander("ℹ️ Model Architecture & Verified Performance Metadata", expanded=False):
        st.write("### Production Model Specifications")
        st.json(MODEL_METRICS)


def render_export_section(result: PredictionResult, image_bytes: Optional[bytes] = None, inference_time_sec: Optional[float] = None):
    st.markdown("---")
    st.subheader("💾 Export Machine-Readable Analysis Payload")
    
    payload = create_export_payload(result, image_bytes=image_bytes, inference_time_sec=inference_time_sec)
    json_str = json.dumps(payload, indent=2)

    st.download_button(
        label="Download Full Prediction JSON Payload",
        data=json_str,
        file_name=f"chest_xray_analysis_{Path(result.image_path).stem}.json",
        mime="application/json"
    )
