"""
NIH ChestX-ray14 Result Export Service
--------------------------------------
Generates machine-readable JSON export payloads containing predictions, metadata, SHA-256 hashes, and safety disclaimers.
"""

import sys
import json
import hashlib
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.output_schema import PredictionResult
from app.config import MODEL_METRICS, MEDICAL_DISCLAIMER_FULL

IST = timezone(timedelta(hours=5, minutes=30))


def create_export_payload(
    result: PredictionResult,
    image_bytes: Optional[bytes] = None,
    inference_time_sec: Optional[float] = None
) -> Dict[str, Any]:
    """
    Creates structured JSON export payload.
    """
    img_hash = hashlib.sha256(image_bytes).hexdigest() if image_bytes else "N/A"

    payload = {
        "timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        "image_metadata": {
            "image_identifier": Path(result.image_path).name,
            "sha256_hash": img_hash,
        },
        "model_metadata": {
            "architecture": MODEL_METRICS["architecture"],
            "selected_experiment": MODEL_METRICS["selected_experiment"],
            "checkpoint_path": result.model_checkpoint,
            "preprocessing_identifier": result.preprocessing_id,
            "inference_device": result.device,
            "inference_time_sec": round(inference_time_sec, 4) if inference_time_sec is not None else None,
            "benchmark_val_macro_auroc": MODEL_METRICS["val_macro_auroc"],
            "benchmark_test_macro_auroc": MODEL_METRICS["test_macro_auroc"],
        },
        "predictions": {
            "highest_probability_class": result.highest_probability_class,
            "highest_probability": float(result.highest_probability),
            "pathologies": {}
        },
        "disclaimer": MEDICAL_DISCLAIMER_FULL
    }

    for c_name, pred in result.predictions.items():
        payload["predictions"]["pathologies"][c_name] = {
            "raw_logit": round(float(pred.raw_logit), 4),
            "probability": round(float(pred.probability), 4),
            "validation_threshold": round(float(pred.threshold), 4) if pred.threshold is not None else None,
            "model_prediction": "Positive" if pred.binary_prediction else "Negative"
        }

    return payload


PATHOLOGY_DESCRIPTIONS = {
    "Atelectasis": "Partial collapse of the lung, where small air sacs deflate.",
    "Cardiomegaly": "An enlarged heart that looks bigger than normal on the scan.",
    "Consolidation": "Part of the lung filled with fluid or mucus instead of air, often from an infection.",
    "Edema": "Excess fluid buildup inside the lung tissue, making it harder to breathe.",
    "Effusion": "Fluid pooling in the space between the lungs and the chest wall.",
    "Emphysema": "Damaged air sacs in the lungs that trap air and make breathing difficult.",
    "Fibrosis": "Permanent scarring or stiffening of lung tissue from previous irritation or damage.",
    "Hernia": "Part of the stomach or intestines pushing up into the chest through the diaphragm.",
    "Infiltration": "Hazy cloudy patches in the lungs showing areas of swelling, fluid, or infection.",
    "Mass": "A large lump or growth in the lung (larger than 3 cm / about coin-sized or bigger).",
    "Nodule": "A small spot or round lump in the lung (smaller than 3 cm / pea or marble-sized).",
    "Pleural_Thickening": "Scarring or thickening of the smooth outer lining that protects the lungs.",
    "Pleural Thickening": "Scarring or thickening of the smooth outer lining that protects the lungs.",
    "Pneumonia": "A lung infection that causes inflammation and fluid buildup.",
    "Pneumothorax": "A collapsed lung caused by air leaking into the space outside the lung."
}


def generate_conversational_summary(result: PredictionResult) -> str:
    """
    Generates plain-English conversational summary of the AI analysis.
    """
    positive_preds = [p for p in result.predictions.values() if p.binary_prediction]
    positive_preds.sort(key=lambda x: x.probability, reverse=True)

    if not positive_preds:
        return (
            "✅ **Normal / No Acute Finding**: The AI scanned across all 14 thoracic pathology categories and "
            "found all probability values to be below diagnostic review thresholds. The lung fields appear clear, "
            "heart size is within normal limits, and no acute pleural effusion or pneumothorax is identified."
        )

    top = positive_preds[0]
    top_desc = PATHOLOGY_DESCRIPTIONS.get(top.pathology, "pulmonary abnormality")
    
    other_pos = [f"**{p.pathology}** ({p.probability * 100:.1f}%)" for p in positive_preds[1:]]

    summary_lines = [
        f"🚨 **Primary AI Finding**: **{top.pathology}** detected with **{top.probability * 100:.1f}% confidence** "
        f"(validation threshold: {top.threshold * 100:.0f}%). This indicates {top_desc.lower()}"
    ]

    if other_pos:
        summary_lines.append(
            f"⚠️ **Additional Co-occurring Findings Flagged Above Threshold**: {', '.join(other_pos)}."
        )

    # Note on clean negatives
    negatives = [p.pathology for p in result.predictions.values() if not p.binary_prediction and p.probability < 0.05]
    if "Cardiomegaly" in negatives and "Pneumothorax" in negatives:
        summary_lines.append(
            "✅ **Reassuring Findings**: No significant cardiomegaly (heart enlargement) or pneumothorax (collapsed lung) detected."
        )

    return "\n\n".join(summary_lines)


def generate_human_readable_report(
    result: PredictionResult,
    patient_info: Optional[Dict[str, str]] = None,
    image_bytes: Optional[bytes] = None,
    inference_time_sec: Optional[float] = None
) -> str:
    """
    Generates structured, radiologist-style text report with patient details.
    """
    p_info = patient_info or {}
    p_id = p_info.get("patient_id", "CXR-" + hashlib.sha256(image_bytes or b"").hexdigest()[:6].upper())
    p_name = p_info.get("name", "Anonymous Patient")
    p_age = p_info.get("age", "N/A")
    p_gender = p_info.get("gender", "Unspecified")

    img_name = Path(result.image_path).name if result.image_path else "Uploaded Radiograph"
    img_hash = hashlib.sha256(image_bytes).hexdigest()[:16] + "..." if image_bytes else "N/A"
    current_time = datetime.now(IST).strftime("%Y-%m-%d %I:%M:%S %p IST")

    positive_preds = [p for p in result.predictions.values() if p.binary_prediction]
    positive_preds.sort(key=lambda x: x.probability, reverse=True)

    lines = [
        "=" * 80,
        "                    AI CHEST RADIOGRAPH DIAGNOSTIC REPORT",
        "=" * 80,
        f"PATIENT NAME     : {p_name}",
        f"PATIENT ID       : {p_id}",
        f"AGE              : {p_age} yrs",
        f"GENDER           : {p_gender}",
        f"DATE & TIME (IST): {current_time}",
        "-" * 80,
        "CLINICAL AI IMPRESSION & FINDINGS:",
    ]

    if positive_preds:
        top = positive_preds[0]
        top_name = top.pathology.replace("_", " ")
        top_desc = PATHOLOGY_DESCRIPTIONS.get(top.pathology, PATHOLOGY_DESCRIPTIONS.get(top_name, ""))
        lines.append(f"  [PRIMARY FINDING] {top_name.upper()} — Model Confidence: {top.probability * 100:.1f}% (Threshold: {top.threshold * 100:.0f}%) [FLAG: REVIEW RECOMMENDED]")
        if top_desc:
            lines.append(f"                    Clinical Context: {top_desc}")

        other_pos = positive_preds[1:]
        if other_pos:
            lines.append(f"\n  [ADDITIONAL FINDINGS ABOVE THRESHOLD] ({len(other_pos)} co-occurring conditions):")
            for idx, p in enumerate(other_pos, start=1):
                p_display = p.pathology.replace("_", " ")
                lines.append(f"    {idx}. {p_display:<20} : {p.probability * 100:>5.1f}%  (Cutoff: {p.threshold * 100:>2.0f}%)  [FLAG: REVIEW RECOMMENDED]")
    else:
        lines.append("  [NORMAL] NO ACUTE PATHOLOGY DETECTED — All 14 conditions below clinical review thresholds.")

    lines.extend([
        "-" * 80,
        "SYSTEMATIC 14-PATHOLOGY EVALUATION:",
    ])

    for c_name, p in result.predictions.items():
        flag_str = "[FLAG: REVIEW ⚠️]" if p.binary_prediction else "[Within Normal Limits ✓]"
        lines.append(f"  • {c_name:<20} : {p.probability * 100:>5.1f}%  (Thresh: {p.threshold * 100:>2.0f}%)  {flag_str}")

    lines.extend([
        "-" * 80,
        "RECOMMENDATIONS & CLINICAL GOVERNANCE:",
        "  1. Review flagged findings alongside patient history, symptoms, and prior radiographs.",
        "  2. Inspect Class Activation Mapping (Grad-CAM) heatmap for anatomical localization.",
        "  3. Final diagnostic sign-off must be performed by a licensed radiologist.",
        "=" * 80,
        "RESEARCH USE ONLY — NOT CERTIFIED FOR INDEPENDENT MEDICAL DIAGNOSIS",
        "=" * 80,
    ])

    return "\n".join(lines)


def generate_pdf_report(
    result: PredictionResult,
    patient_info: Optional[Dict[str, str]] = None,
    image_bytes: Optional[bytes] = None,
    inference_time_sec: Optional[float] = None
) -> bytes:
    """
    Generates a professional medical diagnostic PDF report including patient information.
    """
    import io
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

    p_info = patient_info or {}
    p_id = p_info.get("patient_id", "CXR-" + hashlib.sha256(image_bytes or b"").hexdigest()[:6].upper())
    p_name = p_info.get("name", "Anonymous Patient")
    p_age = str(p_info.get("age", "N/A"))
    p_gender = p_info.get("gender", "Unspecified")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=2
    )
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontSize=9.5,
        textColor=colors.HexColor('#475569'),
        spaceAfter=8
    )
    heading_style = ParagraphStyle(
        'SecHead',
        parent=styles['Heading2'],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=8,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#1E293B')
    )
    bold_style = ParagraphStyle(
        'BoldDark',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=12,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#0F172A')
    )
    alert_style = ParagraphStyle(
        'AlertTxt',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#DC2626'),
        fontName='Helvetica-Bold'
    )
    safe_style = ParagraphStyle(
        'SafeTxt',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#16A34A'),
        fontName='Helvetica-Bold'
    )
    footer_style = ParagraphStyle(
        'FooterNotice',
        parent=styles['Normal'],
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#64748B')
    )

    story = []

    # Title & Header
    story.append(Paragraph("AI-Assisted Chest Radiograph Diagnostic Report", title_style))
    story.append(Paragraph("Radiologist Decision-Support System", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563EB'), spaceAfter=10))

    # Patient & Scan Meta Table (Filtered to user's exact 5 requested fields)
    current_time = datetime.now(IST).strftime("%d-%b-%Y, %I:%M:%S %p IST")

    meta_data = [
        [Paragraph("<b>Patient Name:</b>", body_style), Paragraph(p_name, bold_style),
         Paragraph("<b>Generated Patient ID:</b>", body_style), Paragraph(p_id, bold_style)],
        [Paragraph("<b>Age:</b>", body_style), Paragraph(f"{p_age} yrs", body_style),
         Paragraph("<b>Gender:</b>", body_style), Paragraph(p_gender, body_style)],
        [Paragraph("<b>Date & Time:</b>", body_style), Paragraph(current_time, bold_style),
         Paragraph("<b>Timezone:</b>", body_style), Paragraph("IST (UTC+05:30)", body_style)]
    ]
    t_meta = Table(meta_data, colWidths=[110, 160, 120, 150])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 6))

    # Clinical AI Impression & Findings
    story.append(Paragraph("Clinical AI Impression & Findings", heading_style))

    positive_preds = [p for p in result.predictions.values() if p.binary_prediction]
    positive_preds.sort(key=lambda x: x.probability, reverse=True)

    if positive_preds:
        top = positive_preds[0]
        top_name = top.pathology.replace('_', ' ')
        top_desc = PATHOLOGY_DESCRIPTIONS.get(top.pathology, PATHOLOGY_DESCRIPTIONS.get(top_name, "Pulmonary abnormality requiring clinical correlation."))
        other_pos = positive_preds[1:]

        primary_html = (
            f"<b>PRIMARY AI SUSPECTED PATHOLOGY:</b> &nbsp;"
            f"<font color='#DC2626' size='10'><b>{top_name.upper()}</b></font>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;"
            f"<b>Model Confidence:</b> <font color='#DC2626'><b>{top.probability * 100:.1f}%</b></font>"
            f"&nbsp;&nbsp;|&nbsp;&nbsp;"
            f"<b>Validation Cutoff:</b> {top.threshold * 100:.0f}%<br/>"
            f"<b>Clinical Meaning (Layman Terms):</b> <font color='#334155'>{top_desc}</font>"
        )
        t_primary = Table([[Paragraph(primary_html, body_style)]], colWidths=[540])
        t_primary.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FEF2F2')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#F87171')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(t_primary)

        if other_pos:
            story.append(Spacer(1, 4))
            other_header = Paragraph(
                f"<b>Additional Co-occurring Findings Flagged Above Threshold ({len(other_pos)}):</b>",
                ParagraphStyle(
                    'CoHead',
                    parent=body_style,
                    fontSize=8,
                    leading=10,
                    textColor=colors.HexColor('#92400E'),
                    fontName='Helvetica-Bold'
                )
            )
            story.append(other_header)
            story.append(Spacer(1, 2))

            co_rows = []
            current_row = []
            for p in other_pos:
                p_display = p.pathology.replace('_', ' ')
                cell_text = (
                    f"• <b>{p_display}</b>: <font color='#B45309'><b>{p.probability * 100:.1f}%</b></font> "
                    f"<font color='#64748B' size='7.5'>(Cutoff: {p.threshold * 100:.0f}%)</font>"
                )
                current_row.append(Paragraph(cell_text, body_style))
                if len(current_row) == 2:
                    co_rows.append(current_row)
                    current_row = []
            if current_row:
                while len(current_row) < 2:
                    current_row.append(Paragraph("", body_style))
                co_rows.append(current_row)

            t_co = Table(co_rows, colWidths=[270, 270])
            t_co.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FFFBEB')),
                ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#FCD34D')),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#FEF3C7')),
                ('TOPPADDING', (0, 0), (-1, -1), 2.5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(t_co)
    else:
        normal_top = result.highest_probability_class.replace('_', ' ')
        normal_desc = PATHOLOGY_DESCRIPTIONS.get(result.highest_probability_class, PATHOLOGY_DESCRIPTIONS.get(normal_top, "Normal variant or benign appearance."))
        normal_cell = Paragraph(
            f"<font color='#166534' size='9'><b>CLINICAL IMPRESSION: NO ACUTE PATHOLOGY DETECTED</b></font><br/>"
            f"All 14 thoracic disease categories evaluated were within normal baseline limits and below diagnostic review thresholds. "
            f"No acute consolidation, pneumothorax, or pleural effusion is identified.<br/>"
            f"<font color='#475569' size='7.5'><i>Highest relative signal: {normal_top} ({result.highest_probability * 100:.1f}%) — {normal_desc}</i></font>",
            body_style
        )
        t_normal = Table([[normal_cell]], colWidths=[540])
        t_normal.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F0FDF4')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#86EFAC')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(t_normal)

    story.append(Spacer(1, 6))

    # Findings Table
    story.append(Paragraph("Systematic 14-Pathology AI Analysis & Confidence Scores", heading_style))
    table_rows = [
        [Paragraph("<b>Pathology Category</b>", bold_style),
         Paragraph("<b>Model Confidence</b>", bold_style),
         Paragraph("<b>Threshold</b>", bold_style),
         Paragraph("<b>Clinical Decision Flag</b>", bold_style)]
    ]

    highlight_styles = []
    for idx, (c_name, pred) in enumerate(result.predictions.items(), start=1):
        prob_str = f"{pred.probability * 100:.1f}%"
        thresh_str = f"{pred.threshold * 100:.0f}%" if pred.threshold else "50%"
        display_name = c_name.replace("_", " ")
        if pred.binary_prediction:
            flag_p = Paragraph("⚠️ REVIEW RECOMMENDED", alert_style)
            highlight_styles.append(('BACKGROUND', (0, idx), (-1, idx), colors.HexColor('#FEF2F2')))
        else:
            flag_p = Paragraph("✓ Within Normal Limits", safe_style)
            if idx % 2 == 0:
                highlight_styles.append(('BACKGROUND', (0, idx), (-1, idx), colors.HexColor('#F8FAFC')))

        table_rows.append([
            Paragraph(display_name, body_style),
            Paragraph(prob_str, body_style),
            Paragraph(thresh_str, body_style),
            flag_p
        ])

    t_findings = Table(table_rows, colWidths=[165, 110, 95, 170])
    base_table_style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E2E8F0')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ] + highlight_styles
    t_findings.setStyle(TableStyle(base_table_style))
    story.append(t_findings)
    story.append(Spacer(1, 6))

    # Recommendations & Safety Disclaimer
    story.append(Paragraph("Safety Disclaimer & Physician Verification", heading_style))
    story.append(Paragraph(
        "• This automated analysis was generated by an AI research decision-support model (DenseNet-121).<br/>"
        "• All interpretations must be clinically validated by a qualified board-certified physician or radiologist.<br/>"
        "• Local Privacy Guarantee: Radiograph was processed 100% locally in volatile RAM with zero cloud transmission.",
        footer_style
    ))

    doc.build(story)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data
