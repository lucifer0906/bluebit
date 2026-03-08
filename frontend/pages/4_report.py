"""
AEGIS AI - Page 4: Report Generator
"""

import streamlit as st
import sys
import os
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from core.model_trainer import ModelTrainer
from core.bias_detector import BiasDetector
from core.report_generator import ReportGenerator

st.set_page_config(page_title="Reports | AEGIS AI", page_icon="📄", layout="wide")

st.title("📄 Audit Report Generator")
st.write("Generate comprehensive HTML audit scorecards for download and submission.")

st.divider()

model_name = st.selectbox("Select Model", ["logistic_regression", "random_forest", "xgboost"], index=0)

if st.button("📝 Generate Full Audit Report", type="primary"):
  try:
    with st.spinner("Running audit and generating report..."):
        trainer = ModelTrainer()
        trainer.load_and_prepare_data(include_sensitive=True)
        
        # Try loading from disk first
        try:
            model = trainer.load_model(model_name)
        except Exception:
            trainer.train_all()
            model = trainer.models[model_name]
        
        detector = BiasDetector()
        
        audit = detector.audit_model(
            model=model, model_name=model_name,
            X_train=trainer.X_train, X_test=trainer.X_test,
            y_true=trainer.y_test, sensitive_features=trainer.sensitive_test,
            feature_names=trainer.feature_names,
        )
        
        report_gen = ReportGenerator()
        
        # Generate HTML
        html_content = report_gen.generate_scorecard_html(audit)
        
        # Text summary
        text_summary = report_gen.generate_text_summary(audit)
        
        st.session_state['report'] = {
            'html': html_content,
            'text': text_summary,
            'audit': audit,
            'model_name': model_name,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        st.success("✅ Report generated successfully!")
  except Exception as e:
    st.error(f"⚠️ Report generation failed: {str(e)}. Please ensure models are trained first (go to Upload & Train page).")

if 'report' in st.session_state:
    report = st.session_state['report']
    audit = report['audit']
    verdict = audit['overall_verdict']
    
    st.divider()
    
    # Quick summary
    st.subheader("📋 Audit Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", report['model_name'])
    col2.metric("Grade", verdict['grade'])
    col3.metric("Score", f"{verdict['score']}%")
    col4.metric("Generated", report['timestamp'])
    
    if verdict['passed']:
        st.success(f"✅ **PASSED** — {verdict['label']}")
    else:
        st.error(f"❌ **FAILED** — {verdict['label']}")
    
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["📊 Text Summary", "🌐 HTML Preview", "⬇️ Download"])
    
    with tab1:
        st.subheader("Text Summary")
        st.text(report['text'])
    
    with tab2:
        st.subheader("HTML Report Preview")
        st.components.v1.html(report['html'], height=800, scrolling=True)
    
    with tab3:
        st.subheader("Download Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download HTML Report",
                data=report['html'],
                file_name=f"aegis_ai_audit_{report['model_name']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.html",
                mime="text/html",
                type="primary"
            )
        
        with col2:
            st.download_button(
                label="📥 Download Text Summary",
                data=report['text'],
                file_name=f"aegis_ai_summary_{report['model_name']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain"
            )
        
        st.info("💡 The HTML report includes the full scorecard with metrics, visualizations, and recommendations — ready for hackathon submission.")
