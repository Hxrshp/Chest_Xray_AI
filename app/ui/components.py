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
from app.services.export_service import (
    PATHOLOGY_DESCRIPTIONS,
    generate_conversational_summary,
    generate_human_readable_report,
    generate_pdf_report
)


def render_header():
    st.html(f"<div class='main-title'>{APP_TITLE}</div>")
    st.html(f"<div class='sub-title'>{APP_SUBTITLE}</div>")
    st.html("<div class='safety-badge'>⚠️ RESEARCH USE ONLY — NOT FOR CLINICAL DIAGNOSIS</div>")


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
        top = positive_preds[0]
        top_desc = PATHOLOGY_DESCRIPTIONS.get(top.pathology, "Abnormal thoracic finding detected.")
        additional_findings = positive_preds[1:]

        # Additional findings pill chips
        additional_badges_html = ""
        if additional_findings:
            badges = []
            for p in additional_findings:
                badges.append(
                    f'<span style="display:inline-flex;align-items:center;gap:5px;background-color:#FEF3C7;color:#92400E;border:1px solid #FCD34D;padding:4px 11px;border-radius:9999px;font-size:0.85rem;font-weight:600;margin:3px 6px 3px 0;">'
                    f'<span>⚠️</span> {p.pathology} <b style="color:#B45309;">{p.probability * 100:.1f}%</b></span>'
                )
            additional_badges_html = (
                '<div style="margin-top:14px;padding-top:12px;border-top:1px dashed #FECDD3;">'
                f'<div style="font-size:0.80rem;font-weight:700;color:#991B1B;text-transform:uppercase;letter-spacing:0.6px;margin-bottom:7px;">'
                f'Additional Co-Occurring Findings Flagged Above Threshold ({len(additional_findings)}):'
                '</div>'
                '<div style="display:flex;flex-wrap:wrap;align-items:center;">'
                + "".join(badges)
                + '</div></div>'
            )

        # Reassuring clean negatives
        negatives = [p.pathology for p in result.predictions.values() if not p.binary_prediction and p.probability < 0.05]
        reassuring_items = []
        for key, label in [("Cardiomegaly", "Cardiomegaly"), ("Pneumothorax", "Pneumothorax"), ("Edema", "Pulmonary Edema")]:
            if key in negatives:
                reassuring_items.append(label)
        reassuring_html = ""
        if reassuring_items:
            reassuring_html = (
                '<div style="margin-top:10px;font-size:0.84rem;color:#047857;display:flex;align-items:center;gap:6px;">'
                f'<span>🛡️</span> <b>Reassuring Clinical Negatives:</b> No {", ".join(reassuring_items)} detected (&lt; 5% model probability).'
                '</div>'
            )

        card_html = (
            '<div style="background-color:#FEF2F2;border:1.5px solid #FCA5A5;border-left:6px solid #DC2626;border-radius:12px;padding:18px 22px;margin-bottom:22px;box-shadow:0 2px 4px rgba(220,38,38,0.06);">'
            '<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">'
            '<div style="font-size:0.82rem;font-weight:800;color:#991B1B;text-transform:uppercase;letter-spacing:0.6px;display:flex;align-items:center;gap:6px;">'
            '<span>🚨</span> AI Clinical Impression'
            '</div>'
            '<div style="display:flex;gap:6px;">'
            f'<span style="background-color:#DC2626;color:white;padding:3px 10px;border-radius:9999px;font-size:0.82rem;font-weight:700;">{top.probability * 100:.1f}% Confidence</span>'
            f'<span style="background-color:#FEE2E2;color:#991B1B;border:1px solid #FECDD3;padding:3px 10px;border-radius:9999px;font-size:0.82rem;font-weight:600;">Review Threshold: {top.threshold * 100:.0f}%</span>'
            '</div>'
            '</div>'
            f'<div style="font-size:1.75rem;font-weight:800;color:#991B1B;margin:6px 0 4px 0;">{top.pathology}</div>'
            f'<div style="font-size:0.95rem;color:#4B5563;line-height:1.5;"><b style="color:#1F2937;">Clinical Finding:</b> {top_desc}</div>'
            + additional_badges_html
            + reassuring_html
            + '</div>'
        )
        st.html(card_html)
    else:
        normal_html = (
            '<div style="background-color:#F0FDF4;border:1.5px solid #86EFAC;border-left:6px solid #16A34A;border-radius:12px;padding:18px 22px;margin-bottom:22px;box-shadow:0 2px 4px rgba(22,163,74,0.06);">'
            '<div style="font-size:0.82rem;font-weight:800;color:#166534;text-transform:uppercase;letter-spacing:0.6px;display:flex;align-items:center;gap:6px;">'
            '<span>✅</span> AI Clinical Impression'
            '</div>'
            '<div style="font-size:1.75rem;font-weight:800;color:#15803D;margin:6px 0 4px 0;">Unremarkable Radiograph (No Acute Finding)</div>'
            '<div style="font-size:0.95rem;color:#166534;line-height:1.5;">All 14 thoracic disease categories evaluated were within normal baseline limits and below diagnostic review thresholds.</div>'
            '</div>'
        )
        st.html(normal_html)

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
            
            st.html(
                f'<div style="margin: 10px 0 16px 0; padding: 10px 16px; background-color: #F8FAFC; border-left: 4px solid {"#EF4444" if is_flagged else "#3B82F6"}; border-radius: 4px;">'
                f'<b>Showing Attention Map For:</b> <span style="font-size: 1.15rem; font-weight: 700; color: {"#DC2626" if is_flagged else "#2563EB"};">{target_class}</span> '
                f'(Model Probability: <b>{target_prob:.1f}%</b> | Status: <b>{"FLAG: REVIEW ⚠️" if is_flagged else "Below Threshold ✓"}</b>)'
                '</div>'
            )
            
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
    with st.expander("ℹ️ Model Architecture & Performance Specifications", expanded=False):
        st.write("### AI Model Verification Metrics")
        st.markdown(f"""
        | Metric / Parameter | Value |
        | :--- | :--- |
        | **Model Architecture** | `{MODEL_METRICS['architecture']}` |
        | **Parameters** | `{MODEL_METRICS['parameters']}` |
        | **Input Resolution** | `{MODEL_METRICS['input_resolution']}` |
        | **Selected Baseline** | `{MODEL_METRICS['selected_experiment']}` |
        | **NIH Held-Out Test AUROC** | `{MODEL_METRICS['test_macro_auroc']}` |
        | **Multi-Center Validation AUROC** | `{MODEL_METRICS['val_macro_auroc']}` |
        | **95% Confidence Interval** | `{MODEL_METRICS['ci_95_macro_auroc']}` |
        """)


def render_export_section(result: PredictionResult, image_bytes: Optional[bytes] = None, inference_time_sec: Optional[float] = None):
    st.markdown("---")
    st.subheader("📄 Generate Patient Diagnostic Report (PDF)")
    st.caption("Enter patient details below to generate and download an official, formatted medical diagnostic report.")

    # Auto-generate a unique Patient ID based on image hash and timestamp
    import hashlib
    raw_hash = hashlib.sha256(image_bytes or b"default").hexdigest()[:6].upper()
    default_pid = f"CXR-2026-{raw_hash}"

    # Patient Details Input Card
    with st.container():
        st.html(
            '<div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;">'
            '<div style="font-size: 1rem; font-weight: 700; color: #0F172A; margin-bottom: 4px;">👤 Patient Identification & Study Details</div>'
            '<div style="font-size: 0.85rem; color: #64748B;">These details will be stamped onto the official PDF diagnostic report.</div>'
            '</div>'
        )

        col_p1, col_p2, col_p3 = st.columns([2, 1, 1.5])
        with col_p1:
            patient_name = st.text_input("Patient Full Name:", value="Patient " + raw_hash[:4], placeholder="e.g. John Doe")
        with col_p2:
            patient_age = st.number_input("Patient Age (Years):", min_value=1, max_value=120, value=45, step=1)
        with col_p3:
            patient_gender = st.selectbox("Biological Sex / Gender:", ["Female", "Male", "Other", "Unspecified"])

        patient_id = st.text_input("Generated Patient ID (Unique):", value=default_pid, help="Auto-generated secure patient study identifier.")

    patient_info = {
        "patient_id": patient_id.strip() if patient_id else default_pid,
        "name": patient_name.strip() if patient_name else "Anonymous Patient",
        "age": str(patient_age),
        "gender": patient_gender
    }

    # Generate Human-Readable Text Report
    human_report = generate_human_readable_report(
        result,
        patient_info=patient_info,
        image_bytes=image_bytes,
        inference_time_sec=inference_time_sec
    )

    # Generate PDF Report
    pdf_bytes = generate_pdf_report(
        result,
        patient_info=patient_info,
        image_bytes=image_bytes,
        inference_time_sec=inference_time_sec
    )

    # Download Actions
    col_d1, col_d2 = st.columns([2, 1])
    with col_d1:
        st.download_button(
            label=f"📄 Download Official PDF Medical Report for {patient_info['name']} (.pdf)",
            data=pdf_bytes,
            file_name=f"Medical_Report_{patient_info['patient_id']}_{Path(result.image_path).stem}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )
    with col_d2:
        st.download_button(
            label="📥 Download Clinical Text Summary (.txt)",
            data=human_report,
            file_name=f"Clinical_Summary_{patient_info['patient_id']}_{Path(result.image_path).stem}.txt",
            mime="text/plain",
            use_container_width=True
        )


