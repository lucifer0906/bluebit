"""
AEGIS AI - Page 1: Upload & Train Models
"""

import streamlit as st
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from data.generate_data import generate_hiring_data
from core.model_trainer import ModelTrainer

st.set_page_config(page_title="Upload & Train | AEGIS AI", page_icon="📤", layout="wide")

st.title("📤 Upload & Train Models")
st.write("Generate synthetic hiring data with built-in biases and train ML models for auditing.")

st.divider()

# Section 1: Generate Data
st.header("1️⃣ Generate Synthetic Hiring Dataset")
st.write("This creates a dataset with intentional gender, race, and age biases for testing.")

col1, col2 = st.columns(2)
with col1:
    n_samples = st.slider("Number of candidates", 500, 5000, 2000, step=100)
with col2:
    random_seed = st.number_input("Random seed", value=42)

if st.button("🔄 Generate Dataset", type="primary"):
    with st.spinner("Generating synthetic hiring data..."):
        df = generate_hiring_data(n_samples=n_samples, random_state=int(random_seed))
        st.session_state['dataset'] = df
        st.success(f"✅ Generated {len(df)} candidates!")
        
        # Show statistics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Candidates", len(df))
        col2.metric("Hired", df['hired'].sum())
        col3.metric("Hire Rate", f"{df['hired'].mean()*100:.1f}%")
        col4.metric("Features", len(df.columns) - 2)
        
        # Show sample data
        st.subheader("Sample Data")
        st.dataframe(df.head(20))
        
        # Bias visualization
        st.subheader("📊 Dataset Distribution")
        tab1, tab2, tab3 = st.tabs(["Gender", "Race", "Age"])
        
        with tab1:
            gender_stats = df.groupby('gender')['hired'].agg(['count', 'sum', 'mean']).reset_index()
            gender_stats.columns = ['Gender', 'Total', 'Hired', 'Hire Rate']
            st.dataframe(gender_stats)
            
        with tab2:
            race_stats = df.groupby('race')['hired'].agg(['count', 'sum', 'mean']).reset_index()
            race_stats.columns = ['Race', 'Total', 'Hired', 'Hire Rate']
            st.dataframe(race_stats)
            
        with tab3:
            df['age_group'] = pd.cut(df['age'], bins=[20, 30, 40, 50, 60], labels=['20-30', '30-40', '40-50', '50-60'])
            age_stats = df.groupby('age_group')['hired'].agg(['count', 'sum', 'mean']).reset_index()
            age_stats.columns = ['Age Group', 'Total', 'Hired', 'Hire Rate']
            st.dataframe(age_stats)

st.divider()

# Section 2: Train Models
st.header("2️⃣ Train ML Models")
st.write("Train multiple models to compare bias levels.")

include_sensitive = st.checkbox("Include sensitive features (gender, race) in model", value=True,
                                help="Including sensitive features will make the model biased - this is for testing purposes.")

selected_models = st.multiselect(
    "Select models to train",
    ["logistic_regression", "random_forest", "xgboost"],
    default=["logistic_regression", "random_forest", "xgboost"]
)

if st.button("🚀 Train Models", type="primary"):
    with st.spinner("Training models..."):
        trainer = ModelTrainer()
        trainer.load_and_prepare_data(include_sensitive=include_sensitive)
        
        progress = st.progress(0)
        
        for i, model_name in enumerate(selected_models):
            st.write(f"Training {model_name}...")
            if model_name == "logistic_regression":
                trainer.train_logistic_regression()
            elif model_name == "random_forest":
                trainer.train_random_forest()
            elif model_name == "xgboost":
                trainer.train_xgboost()
            progress.progress((i + 1) / len(selected_models))
        
        trainer.save_models()
        st.session_state['trainer'] = trainer
        st.session_state['trained_models'] = selected_models
        
        st.success(f"✅ Trained {len(selected_models)} models!")
        
        # Show model performance
        st.subheader("Model Performance")
        for model_name in selected_models:
            model = trainer.models[model_name]
            y_pred, y_prob = trainer.get_predictions(model_name=model_name)
            accuracy = (y_pred == trainer.y_test).mean()
            st.write(f"**{model_name}**: Accuracy = {accuracy:.4f}")
