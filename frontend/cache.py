"""
AEGIS AI - Cached resource helpers for Streamlit pages.
Avoids redundant data loading and model loading across page interactions.
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.model_trainer import ModelTrainer


@st.cache_resource
def get_cached_trainer(include_sensitive=True):
    """Load and prepare data once, cache across reruns."""
    trainer = ModelTrainer()
    trainer.load_and_prepare_data(include_sensitive=include_sensitive)
    return trainer


@st.cache_resource
def get_cached_model(model_name, include_sensitive=True):
    """Load a pre-trained model from disk, or train if missing."""
    trainer = get_cached_trainer(include_sensitive)
    try:
        return trainer.load_model(model_name)
    except Exception:
        if model_name == 'logistic_regression':
            trainer.train_logistic_regression()
        elif model_name == 'random_forest':
            trainer.train_random_forest()
        elif model_name == 'xgboost':
            trainer.train_xgboost()
        trainer.save_models()
        return trainer.models[model_name]
