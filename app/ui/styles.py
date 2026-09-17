"""
Streamlit UI Custom Styles
--------------------------
Clean, research-oriented medical CSS styling for radiologist decision-support dashboard elements.
"""

CUSTOM_CSS = """
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 0.8rem;
    }
    .safety-badge {
        display: inline-block;
        background-color: #FEF2F2;
        border: 1px solid #FCA5A5;
        color: #991B1B;
        padding: 0.4rem 0.8rem;
        border-radius: 0.375rem;
        font-weight: 700;
        font-size: 0.85rem;
        margin-bottom: 1.2rem;
    }
    .preprocessing-box {
        background-color: #F1F5F9;
        border: 1px solid #CBD5E1;
        padding: 0.75rem;
        border-radius: 0.375rem;
        font-size: 0.88rem;
        color: #334155;
    }
    .finding-card-pos {
        background-color: #FFF7ED;
        border-left: 4px solid #F97316;
        padding: 0.85rem;
        border-radius: 0.375rem;
        margin-bottom: 0.5rem;
    }
    .finding-card-neg {
        background-color: #F8FAFC;
        border-left: 4px solid #94A3B8;
        padding: 0.85rem;
        border-radius: 0.375rem;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.5rem;
        color: #0F172A;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #64748B;
        font-weight: 600;
    }
    /* Responsive X-ray Image Constraints — Fits viewport without giant scrolling */
    div[data-testid="stImage"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important;
        margin: 0 auto !important;
    }
    div[data-testid="stImage"] img {
        max-height: 370px !important;
        max-width: 100% !important;
        width: auto !important;
        height: auto !important;
        object-fit: contain !important;
        margin: 0 auto !important;
        display: block !important;
        border-radius: 8px !important;
        border: 1px solid #CBD5E1 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
    }
    div[data-testid="stImageCaption"] {
        text-align: center !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        color: #334155 !important;
        margin-top: 6px !important;
    }
</style>
"""
