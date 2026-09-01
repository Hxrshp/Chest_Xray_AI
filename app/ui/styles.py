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
</style>
"""
