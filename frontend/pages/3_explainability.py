"""
AEGIS AI - Page 3: Explainability Dashboard
"""

import streamlit as st
import sys
import os
import plotly.graph_objects as go
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from core.model_trainer import ModelTrainer
from core.explainability import ExplainabilityEngine

st.set_page_config(page_title="Explainability | AEGIS AI", page_icon="🧠", layout="wide")

st.title("🧠 Model Explainability")
st.write("Understand WHY the model makes its decisions using SHAP analysis.")

st.divider()

model_name = st.selectbox("Select Model", ["logistic_regression", "random_forest", "xgboost"], index=1)

if st.button("🔬 Analyze Model Decisions", type="primary"):
  try:
    with st.spinner("Computing SHAP values — this may take a moment..."):
        trainer = ModelTrainer()
        trainer.load_and_prepare_data(include_sensitive=True)
        
        # Try loading from disk first
        try:
            model = trainer.load_model(model_name)
        except Exception:
            trainer.train_all()
            model = trainer.models[model_name]
        
        engine = ExplainabilityEngine(model, trainer.X_train, trainer.feature_names)
        engine.initialize_shap(trainer.X_test[:100])
        
        st.session_state['explainability'] = {
            'engine': engine,
            'trainer': trainer,
            'model_name': model_name,
            'model': model
        }
        st.success("✅ SHAP analysis complete!")
  except Exception as e:
    st.error(f"⚠️ Analysis failed: {str(e)}. Please ensure models are trained first (go to Upload & Train page).")

if 'explainability' in st.session_state:
    data = st.session_state['explainability']
    engine = data['engine']
    trainer = data['trainer']
    
    tab1, tab2, tab3 = st.tabs(["📊 Feature Importance", "👤 Individual Explanations", "⚠️ Bias in Features"])
    
    with tab1:
        st.subheader("Global Feature Importance")
        st.write("Which features have the most influence on hiring decisions?")
        
        importance = engine.get_feature_importance(trainer.X_test[:100])
        
        if importance:
            features = list(importance.keys())
            values = list(importance.values())
            
            # Color sensitive features differently
            sensitive = ['gender', 'race', 'age']
            colors = ['#ef4444' if any(s in f.lower() for s in sensitive) else '#3b82f6' 
                       for f in features]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=values, y=features,
                    orientation='h',
                    marker_color=colors
                )
            ])
            fig.update_layout(
                title="Feature Importance (Mean |SHAP|)",
                xaxis_title="Mean |SHAP value|",
                yaxis=dict(autorange="reversed"),
                template="plotly_dark",
                height=max(400, len(features) * 35)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("🔴 **Red bars** = sensitive/protected features | 🔵 **Blue bars** = non-sensitive features")
            
            # Summary table
            import pandas as pd
            df_imp = pd.DataFrame({
                'Feature': features,
                'SHAP Importance': [f"{v:.4f}" for v in values],
                'Sensitive': ['⚠️ Yes' if any(s in f.lower() for s in sensitive) else 'No' for f in features]
            })
            st.dataframe(df_imp, use_container_width=True)
    
    with tab2:
        st.subheader("Individual Candidate Explanations")
        st.write("See why a specific candidate was hired or rejected.")
        
        n_samples = len(trainer.X_test)
        candidate_idx = st.slider("Select Candidate Index", 0, n_samples - 1, 0)
        
        if st.button("Explain This Decision"):
            with st.spinner("Generating explanation..."):
                explanation = engine.explain_individual(
                    trainer.X_test[candidate_idx],
                    candidate_id=candidate_idx
                )
                
                if explanation:
                    # Prediction info
                    prediction = explanation.get('prediction', 'N/A')
                    st.metric("Model Decision", "HIRED ✅" if prediction == 1 else "REJECTED ❌")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**🟢 Top Positive Factors (pushing toward HIRE):**")
                        for factor in explanation.get('top_positive_factors', []):
                            st.write(f"  • **{factor['feature']}**: +{factor['shap_value']:.4f}")
                    
                    with col2:
                        st.write("**🔴 Top Negative Factors (pushing toward REJECT):**")
                        for factor in explanation.get('top_negative_factors', []):
                            st.write(f"  • **{factor['feature']}**: {factor['shap_value']:.4f}")
                    
                    # Waterfall-style chart
                    all_factors = explanation.get('top_positive_factors', []) + explanation.get('top_negative_factors', [])
                    if all_factors:
                        all_factors.sort(key=lambda x: abs(x['shap_value']), reverse=True)
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=[f['shap_value'] for f in all_factors],
                                y=[f['feature'] for f in all_factors],
                                orientation='h',
                                marker_color=['#22c55e' if f['shap_value'] > 0 else '#ef4444' for f in all_factors]
                            )
                        ])
                        fig.update_layout(
                            title=f"Candidate #{candidate_idx} — Decision Factors",
                            xaxis_title="SHAP Contribution",
                            template="plotly_dark",
                            height=max(300, len(all_factors) * 35)
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Bias Detection in Feature Usage")
        st.write("Are sensitive features disproportionately influencing decisions?")
        
        bias_analysis = engine.detect_bias_in_explanations(
            trainer.X_test[:100], trainer.sensitive_test.iloc[:100]
        )
        
        if bias_analysis:
            flagged = bias_analysis.get('flagged_sensitive_features', [])
            
            if bias_analysis.get('bias_detected', False):
                st.error(f"⚠️ Found {len([f for f in flagged if f.get('is_concerning')])} potentially biased feature(s)!")
                for b in flagged:
                    if b.get('is_concerning', False):
                        st.warning(f"""
                        **Feature:** {b['feature']}  
                        **Importance:** {b['importance']:.4f} ({b['percentage']:.1f}%)  
                        **Threshold:** 10%  
                        **Risk:** This sensitive feature heavily influences model decisions.
                        """)
            else:
                st.success("✅ No sensitive features exceed the 10% importance threshold.")
            
            st.write("**All Sensitive Feature Importances:**")
            for f in flagged:
                pct = f['percentage']
                color = "🔴" if pct > 10 else "🟡" if pct > 5 else "🟢"
                st.write(f"{color} **{f['feature']}**: {pct:.1f}%")
