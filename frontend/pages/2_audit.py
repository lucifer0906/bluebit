"""
AEGIS AI - Page 2: Bias Audit Dashboard
"""

import streamlit as st
import sys
import os
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from core.model_trainer import ModelTrainer
from core.bias_detector import BiasDetector

st.set_page_config(page_title="Bias Audit | AEGIS AI", page_icon="🔍", layout="wide")

st.title("🔍 Bias Audit Dashboard")
st.write("Run comprehensive fairness audits on trained hiring models.")

st.divider()

# Model selection
model_name = st.selectbox(
    "Select model to audit",
    ["logistic_regression", "random_forest", "xgboost"],
    index=0
)

include_sensitive = st.checkbox("Model includes sensitive features", value=True)

if st.button("🔍 Run Audit", type="primary"):
  try:
    with st.spinner("Running comprehensive bias audit..."):
        # Load pre-trained models or train if needed
        trainer = ModelTrainer()
        trainer.load_and_prepare_data(include_sensitive=include_sensitive)
        
        # Try loading from disk first
        try:
            model = trainer.load_model(model_name)
        except Exception:
            trainer.train_all()
            model = trainer.models[model_name]
        
        detector = BiasDetector()
        
        audit = detector.audit_model(
            model=model,
            model_name=model_name,
            X_train=trainer.X_train,
            X_test=trainer.X_test,
            y_true=trainer.y_test,
            sensitive_features=trainer.sensitive_test,
            feature_names=trainer.feature_names,
        )
        
        st.session_state['current_audit'] = audit
        
        # Overall Verdict
        verdict = audit['overall_verdict']
        
        st.subheader("📋 Overall Verdict")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Fairness Score", f"{verdict['score']}%")
        col2.metric("Grade", verdict['grade'])
        col3.metric("Tests Passed", f"{verdict['tests_passed']}/{verdict['total_tests']}")
        col4.metric("Accuracy", f"{audit['performance']['accuracy']:.4f}")
        
        if verdict['passed']:
            st.success(f"✅ Model PASSES fairness audit — {verdict['label']}")
        else:
            st.error(f"❌ Model FAILS fairness audit — {verdict['label']}")
        
        st.divider()
        
        # Metrics Scorecard
        st.subheader("📊 Fairness Metrics Scorecard")
        
        metrics = audit['fairness']['metrics']
        
        for key, result in metrics.items():
            passed = result.get('passed', False)
            icon = "✅" if passed else "❌"
            
            with st.expander(f"{icon} {result.get('metric', key)} — {'PASS' if passed else 'FAIL'}", expanded=not passed):
                st.write(f"**Description:** {result.get('description', '')}")
                
                if 'selection_rates' in result:
                    st.write(f"**Selection Rates:** {result['selection_rates']}")
                    
                    # Bar chart
                    rates = result['selection_rates']
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(rates.keys()),
                            y=list(rates.values()),
                            marker_color=['#22c55e' if v >= result.get('threshold', 0.8) * max(rates.values()) 
                                         else '#ef4444' for v in rates.values()]
                        )
                    ])
                    fig.update_layout(
                        title=f"Selection Rates by {result.get('attribute', '')}",
                        yaxis_title="Selection Rate",
                        template="plotly_dark",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                if 'true_positive_rates' in result:
                    st.write(f"**True Positive Rates:** {result['true_positive_rates']}")
                    st.write(f"**Max Difference:** {result.get('max_difference', 'N/A')}")
                    
                    rates = result['true_positive_rates']
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(rates.keys()),
                            y=list(rates.values()),
                            marker_color=['#22c55e' if abs(v - max(rates.values())) <= 0.1 
                                         else '#ef4444' for v in rates.values()]
                        )
                    ])
                    fig.update_layout(
                        title=f"True Positive Rates by {result.get('attribute', '')}",
                        yaxis_title="TPR",
                        template="plotly_dark",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                if 'brier_score' in result:
                    st.write(f"**Brier Score:** {result['brier_score']}")
                
                if 'score' in result:
                    st.write(f"**Transparency Score:** {result['score']}/100")
                
                if 'sensitive_features_used' in result:
                    used = result['sensitive_features_used']
                    if used:
                        st.warning(f"⚠️ Sensitive features used: {', '.join(used)}")
                    else:
                        st.success("No sensitive features used directly.")
        
        st.divider()
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        for rec in audit['recommendations']:
            severity = rec.get('severity', 'INFO')
            if severity == 'CRITICAL':
                st.error(f"🔴 **[{severity}] {rec['category']}**\n\n{rec['finding']}\n\n**Recommendation:** {rec['recommendation']}")
            elif severity == 'HIGH':
                st.warning(f"🟠 **[{severity}] {rec['category']}**\n\n{rec['finding']}\n\n**Recommendation:** {rec['recommendation']}")
            else:
                st.info(f"🟢 **[{severity}] {rec['category']}**\n\n{rec['recommendation']}")
  except Exception as e:
    st.error(f"⚠️ Audit failed: {str(e)}. Please ensure models are trained first (go to Upload & Train page).")

st.divider()

# Multi-model comparison
st.subheader("🔄 Compare Multiple Models")
if st.button("Compare All Models"):
  try:
    with st.spinner("Auditing all models for comparison..."):
        trainer = ModelTrainer()
        trainer.load_and_prepare_data(include_sensitive=True)
        
        detector = BiasDetector()
        
        comparison_data = []
        for mname in ['logistic_regression', 'random_forest', 'xgboost']:
            # Try loading from disk first
            try:
                model = trainer.load_model(mname)
            except Exception:
                trainer.train_all()
                model = trainer.models[mname]
            
            audit = detector.audit_model(
                model=model, model_name=mname,
                X_train=trainer.X_train, X_test=trainer.X_test,
                y_true=trainer.y_test, sensitive_features=trainer.sensitive_test,
                feature_names=trainer.feature_names,
            )
            verdict = audit['overall_verdict']
            comparison_data.append({
                'Model': mname,
                'Accuracy': audit['performance']['accuracy'],
                'Fairness Score': verdict['score'],
                'Grade': verdict['grade'],
                'Passed': verdict['passed'],
                'Tests Passed': verdict['tests_passed'],
                'Tests Failed': verdict['tests_failed'],
            })
        
        import pandas as pd
        df_compare = pd.DataFrame(comparison_data)
        st.dataframe(df_compare, use_container_width=True)
        
        # Comparison chart
        fig = go.Figure(data=[
            go.Bar(name='Accuracy', x=df_compare['Model'], y=df_compare['Accuracy'] * 100),
            go.Bar(name='Fairness Score', x=df_compare['Model'], y=df_compare['Fairness Score']),
        ])
        fig.update_layout(
            title="Model Comparison: Accuracy vs Fairness",
            yaxis_title="Score (%)",
            barmode='group',
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
  except Exception as e:
    st.error(f"⚠️ Comparison failed: {str(e)}. Please ensure models are trained first.")
