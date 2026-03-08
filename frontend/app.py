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

st.markdown("### 🚀 Quick Demo")
st.write("Click below to instantly run a full bias audit on our pre-trained Random Forest model:")

if st.button("⚡ Run Quick Demo Audit", type="primary"):
    with st.spinner("Running bias audit on Random Forest..."):
        from frontend.cache import get_cached_trainer, get_cached_model
        from core.bias_detector import BiasDetector

        trainer = get_cached_trainer(include_sensitive=True)
        model = get_cached_model("random_forest", include_sensitive=True)
        detector = BiasDetector()
        audit = detector.audit_model(
            model=model, model_name="random_forest",
            X_train=trainer.X_train, X_test=trainer.X_test,
            y_true=trainer.y_test, sensitive_features=trainer.sensitive_test,
            feature_names=trainer.feature_names,
        )
        verdict = audit['overall_verdict']
        st.session_state['quick_demo_audit'] = audit

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Fairness Score", f"{verdict['score']}%")
    col_b.metric("Grade", verdict['grade'])
    col_c.metric("Tests Passed", f"{verdict['tests_passed']}/{verdict['total_tests']}")
    col_d.metric("Accuracy", f"{audit['performance']['accuracy']:.4f}")

    if verdict['passed']:
        st.success(f"✅ Model PASSES fairness audit — {verdict['label']}")
    else:
        st.error(f"❌ Model FAILS fairness audit — {verdict['label']}")

    st.info("👈 Navigate to **Audit**, **Explainability**, or **Report** pages in the sidebar for deeper analysis.")

st.divider()

st.markdown("### 📖 How to Use")
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
