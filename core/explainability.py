"""
AEGIS AI - Explainability Module
Provides SHAP and LIME explanations for model decisions.
"""

import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class ExplainabilityEngine:
    """Generate explanations for ML model predictions."""
    
    def __init__(self, model, X_train: np.ndarray, feature_names: List[str]):
        """
        Args:
            model: Trained sklearn-compatible model
            X_train: Training data for background distribution
            feature_names: List of feature names
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.shap_explainer = None
        self.shap_values = None
        
    def initialize_shap(self, X_data: np.ndarray = None):
        """Initialize SHAP explainer and compute SHAP values."""
        model_type = type(self.model).__name__
        
        if model_type in ['RandomForestClassifier', 'GradientBoostingClassifier', 'XGBClassifier']:
            self.shap_explainer = shap.TreeExplainer(self.model)
        else:
            # Use KernelExplainer for other model types
            background = shap.sample(self.X_train, min(100, len(self.X_train)))
            self.shap_explainer = shap.KernelExplainer(
                self.model.predict_proba, background
            )
        
        if X_data is not None:
            self.shap_values = self.shap_explainer.shap_values(X_data)
        
        return self.shap_explainer
    
    def get_feature_importance(self, X_data: np.ndarray) -> Dict[str, float]:
        """
        Get global feature importance using SHAP values.
        
        Returns:
            Dictionary mapping feature names to importance scores.
        """
        if self.shap_explainer is None:
            self.initialize_shap(X_data)
        
        shap_values = self.shap_explainer.shap_values(X_data)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Older SHAP: list of arrays per class
            shap_vals = np.array(shap_values[1])
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # Newer SHAP: (n_samples, n_features, n_classes)
            shap_vals = shap_values[:, :, 1]
        else:
            shap_vals = np.array(shap_values)
        
        # Mean absolute SHAP values per feature
        importance = np.abs(shap_vals).mean(axis=0)
        
        importance_dict = {}
        for i, name in enumerate(self.feature_names):
            val = importance[i]
            if hasattr(val, '__len__'):
                val = float(np.mean(np.abs(val)))
            importance_dict[name] = round(float(val), 4)
        
        # Sort by importance
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        return importance_dict
    
    def explain_individual(self, X_instance: np.ndarray, 
                          candidate_id: int = None) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP values.
        
        Args:
            X_instance: Single data point (1D or 2D array)
            candidate_id: Optional identifier
            
        Returns:
            Dictionary with explanation details.
        """
        if self.shap_explainer is None:
            self.initialize_shap()
        
        if X_instance.ndim == 1:
            X_instance = X_instance.reshape(1, -1)
        
        shap_values = self.shap_explainer.shap_values(X_instance)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_vals = np.array(shap_values[1][0])
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_vals = shap_values[0, :, 1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
            shap_vals = shap_values[0]
        else:
            shap_vals = np.array(shap_values)
        
        # Get prediction
        prediction = self.model.predict(X_instance)[0]
        probability = self.model.predict_proba(X_instance)[0]
        
        # Build explanation
        feature_contributions = []
        for i, name in enumerate(self.feature_names):
            feature_contributions.append({
                'feature': name,
                'value': round(float(X_instance[0][i]), 4),
                'shap_value': round(float(shap_vals[i]), 4),
                'direction': 'positive' if shap_vals[i] > 0 else 'negative',
                'impact': 'Increases hiring chance' if shap_vals[i] > 0 
                         else 'Decreases hiring chance'
            })
        
        # Sort by absolute SHAP value
        feature_contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        # Generate human-readable explanation
        top_positive = [f for f in feature_contributions if f['direction'] == 'positive'][:3]
        top_negative = [f for f in feature_contributions if f['direction'] == 'negative'][:3]
        
        explanation_text = f"Prediction: {'HIRED' if prediction == 1 else 'NOT HIRED'} "
        explanation_text += f"(Confidence: {max(probability)*100:.1f}%)\n\n"
        
        if top_positive:
            explanation_text += "Factors FAVORING this candidate:\n"
            for f in top_positive:
                explanation_text += f"  • {f['feature']}: {f['value']} (impact: +{f['shap_value']:.3f})\n"
        
        if top_negative:
            explanation_text += "\nFactors AGAINST this candidate:\n"
            for f in top_negative:
                explanation_text += f"  • {f['feature']}: {f['value']} (impact: {f['shap_value']:.3f})\n"
        
        return {
            'candidate_id': candidate_id,
            'prediction': int(prediction),
            'probability': {
                'not_hired': round(float(probability[0]), 4),
                'hired': round(float(probability[1]), 4)
            },
            'feature_contributions': feature_contributions,
            'explanation_text': explanation_text,
            'top_positive_factors': top_positive[:3],
            'top_negative_factors': top_negative[:3],
        }
    
    def detect_bias_in_explanations(self, X_data: np.ndarray, 
                                     sensitive_features: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze SHAP values to detect if sensitive features have outsized influence.
        
        Returns:
            Bias detection results with flagged features.
        """
        importance = self.get_feature_importance(X_data)
        
        sensitive_attrs = {'gender', 'race', 'age', 'sex', 'ethnicity'}
        
        flagged = []
        total_importance = sum(importance.values())
        
        for feature, imp in importance.items():
            if feature.lower() in sensitive_attrs:
                pct = (imp / total_importance) * 100 if total_importance > 0 else 0
                flagged.append({
                    'feature': feature,
                    'importance': imp,
                    'percentage': round(pct, 2),
                    'is_concerning': pct > 10,  # Flag if > 10% of total importance
                })
        
        return {
            'feature_importance': importance,
            'flagged_sensitive_features': flagged,
            'bias_detected': any(f['is_concerning'] for f in flagged),
            'recommendation': 'Consider removing or reducing the influence of flagged '
                            'sensitive features to reduce bias.' if flagged else
                            'No sensitive features detected in model inputs.'
        }
