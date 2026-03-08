"""
AEGIS AI - Streamlit Dashboard
Main entry point for the frontend application.
"""

import streamlit as st
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="AEGIS AI — Ethical AI Auditor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        color: #00d4ff;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2em;
        text-align: center;
        color: #8899aa;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #0a1628, #1a2744);
        border: 1px solid #00d4ff33;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .stMetric {
        background: #111827;
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #1f2937;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">⚖️ AEGIS AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ethical AI Auditing Framework for Hiring System Bias Detection</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#666;">Bluebit Hackathon 2026 | PS9 — Jedi Code Compliance System | Team MISAL PAV</p>', unsafe_allow_html=True)

st.divider()

# Main page content
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🔍 Detect Bias")
    st.write("Upload or train ML models and detect hidden biases in hiring decisions across gender, race, and age groups.")

with col2:
    st.markdown("### 📊 Analyze Fairness")
    st.write("Run comprehensive fairness audits with Demographic Parity, Equal Opportunity, Calibration, and more.")

with col3:
    st.markdown("### 📋 Generate Reports")
    st.write("Get visual scorecards, SHAP explanations, and actionable recommendations to fix bias.")

st.divider()

st.markdown("### 🚀 Getting Started")
st.write("Use the sidebar to navigate between pages:")
st.markdown("""
1. **📤 Upload & Train** — Generate synthetic data and train ML models
2. **🔍 Audit** — Run bias audits and view fairness metrics
3. **🧠 Explainability** — SHAP-based feature importance and decision explanations
4. **📄 Report** — Generate and download the full audit report
""")

# Quick stats
st.divider()
st.markdown("### ℹ️ About This Project")

st.info("""
**Problem:** AI-powered hiring systems can encode biases related to gender, race, and age.

**Solution:** AEGIS AI audits ML models for ethical compliance using:
- **Demographic Parity** — Equal selection rates across groups
- **Equal Opportunity** — Equal true positive rates  
- **Calibration** — Predicted probabilities match reality
- **SHAP Explainability** — Understand WHY the model makes each decision
- **Privacy Checks** — Detect use of sensitive attributes

**Tech Stack:** Python, Scikit-learn, XGBoost, SHAP, Streamlit, FastAPI, Plotly
""")
