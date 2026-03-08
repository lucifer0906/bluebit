"""
AEGIS AI - Bias Detector
Main orchestrator for the ethical AI auditing pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

from core.fairness_metrics import FairnessMetrics
from core.explainability import ExplainabilityEngine


class BiasDetector:
    """Main bias detection and auditing engine."""
    
    def __init__(self):
        self.models = {}
        self.audit_results = {}
        
    def load_model(self, model_path: str, model_name: str = None):
        """Load a trained model from file."""
        model = joblib.load(model_path)
        name = model_name or os.path.basename(model_path).replace('.joblib', '')
        self.models[name] = model
        return model
    
    def audit_model(self, model, model_name: str,
                    X_train: np.ndarray, X_test: np.ndarray,
                    y_true: np.ndarray, 
                    sensitive_features: pd.DataFrame,
                    feature_names: List[str],
                    sensitive_attributes: List[str] = None) -> Dict[str, Any]:
        """
        Perform a complete bias audit on a model.
        
        Args:
            model: Trained ML model
            model_name: Name identifier for the model
            X_train: Training data (for SHAP background)
            X_test: Test data
            y_true: True labels for test data
            sensitive_features: DataFrame with sensitive attribute columns
            feature_names: List of feature names
            sensitive_attributes: Which attributes to audit (default: all)
            
        Returns:
            Comprehensive audit report.
        """
        if sensitive_attributes is None:
            sensitive_attributes = list(sensitive_features.columns)
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Model performance
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Fairness metrics
        fairness = FairnessMetrics(y_true, y_pred, y_prob, sensitive_features)
        fairness_results = fairness.run_full_audit(
            model=model,
            feature_names=feature_names,
            sensitive_attributes=sensitive_attributes
        )
        
        # Explainability
        explainer = ExplainabilityEngine(model, X_train, feature_names)
        explainer.initialize_shap(X_test[:100])  # Use subset for speed
        
        feature_importance = explainer.get_feature_importance(X_test[:100])
        bias_in_features = explainer.detect_bias_in_explanations(
            X_test[:100], sensitive_features.iloc[:100]
        )
        
        # Sample individual explanations
        sample_explanations = []
        for i in range(min(5, len(X_test))):
            explanation = explainer.explain_individual(X_test[i])
            sample_explanations.append(explanation)
        
        # Compile complete audit
        audit = {
            'model_name': model_name,
            'model_type': type(model).__name__,
            'performance': {
                'accuracy': round(accuracy, 4),
                'classification_report': report,
            },
            'fairness': fairness_results,
            'explainability': {
                'feature_importance': feature_importance,
                'bias_in_features': bias_in_features,
                'sample_explanations': sample_explanations,
            },
            'recommendations': self._generate_recommendations(fairness_results, bias_in_features),
            'overall_verdict': self._compute_verdict(fairness_results),
        }
        
        self.audit_results[model_name] = audit
        return audit
    
    def _generate_recommendations(self, fairness_results: Dict, 
                                   bias_analysis: Dict) -> List[Dict[str, str]]:
        """Generate actionable recommendations based on audit findings."""
        recommendations = []
        
        metrics = fairness_results.get('metrics', {})
        
        # Check demographic parity
        for key, result in metrics.items():
            if 'demographic_parity' in key and not result.get('passed', True):
                attr = result.get('attribute', 'unknown')
                rates = result.get('selection_rates', {})
                recommendations.append({
                    'severity': 'HIGH',
                    'category': 'Demographic Parity',
                    'finding': f"Unequal selection rates across {attr} groups: {rates}",
                    'recommendation': f"Re-train the model without {attr} as a feature, "
                                    f"or apply post-processing calibration to equalize rates.",
                    'impact': 'Candidates from certain groups are systematically disadvantaged.'
                })
        
        # Check equal opportunity
        for key, result in metrics.items():
            if 'equal_opportunity' in key and not result.get('passed', True):
                attr = result.get('attribute', 'unknown')
                tpr = result.get('true_positive_rates', {})
                recommendations.append({
                    'severity': 'HIGH',
                    'category': 'Equal Opportunity',
                    'finding': f"Unequal true positive rates across {attr}: {tpr}",
                    'recommendation': f"Qualified candidates from underrepresented {attr} groups "
                                    f"are being rejected at higher rates. Consider threshold adjustment.",
                    'impact': 'Qualified candidates are being unfairly rejected.'
                })
        
        # Check privacy
        privacy = metrics.get('privacy', {})
        if privacy and not privacy.get('passed', True):
            sensitive_used = privacy.get('sensitive_features_used', [])
            recommendations.append({
                'severity': 'CRITICAL',
                'category': 'Privacy',
                'finding': f"Sensitive features used directly: {sensitive_used}",
                'recommendation': "Remove sensitive attributes from model input features. "
                                "Use fairness-aware training if demographic information is needed.",
                'impact': 'Direct use of protected attributes in decisions is illegal in many jurisdictions.'
            })
        
        # Check feature importance bias
        if bias_analysis.get('bias_detected', False):
            flagged = bias_analysis.get('flagged_sensitive_features', [])
            for f in flagged:
                if f.get('is_concerning', False):
                    recommendations.append({
                        'severity': 'HIGH',
                        'category': 'Feature Bias',
                        'finding': f"Feature '{f['feature']}' has {f['percentage']}% influence on decisions",
                        'recommendation': f"The feature '{f['feature']}' has outsized influence. "
                                        f"Consider removing it or using adversarial debiasing.",
                        'impact': 'Model decisions are significantly influenced by sensitive attributes.'
                    })
        
        if not recommendations:
            recommendations.append({
                'severity': 'INFO',
                'category': 'Overall',
                'finding': 'No major bias issues detected.',
                'recommendation': 'Continue monitoring model fairness regularly.',
                'impact': 'The model appears to be making fair decisions.'
            })
        
        return recommendations
    
    def _compute_verdict(self, fairness_results: Dict) -> Dict[str, Any]:
        """Compute overall audit verdict."""
        score = fairness_results.get('fairness_score', 0)
        
        if score >= 90:
            grade = 'A'
            label = 'Excellent'
            color = 'green'
        elif score >= 75:
            grade = 'B'
            label = 'Good'
            color = 'lightgreen'
        elif score >= 60:
            grade = 'C'
            label = 'Acceptable'
            color = 'yellow'
        elif score >= 40:
            grade = 'D'
            label = 'Poor'
            color = 'orange'
        else:
            grade = 'F'
            label = 'Failing'
            color = 'red'
        
        return {
            'score': score,
            'grade': grade,
            'label': label,
            'color': color,
            'passed': fairness_results.get('overall_passed', False),
            'tests_passed': fairness_results.get('pass_count', 0),
            'tests_failed': fairness_results.get('fail_count', 0),
            'total_tests': fairness_results.get('total_tests', 0),
        }
    
    def compare_models(self) -> Dict[str, Any]:
        """Compare audit results across multiple models."""
        if len(self.audit_results) < 2:
            return {'message': 'Need at least 2 models to compare.'}
        
        comparison = {
            'models': {},
            'best_model': None,
            'best_score': 0,
        }
        
        for name, audit in self.audit_results.items():
            verdict = audit.get('overall_verdict', {})
            comparison['models'][name] = {
                'accuracy': audit['performance']['accuracy'],
                'fairness_score': verdict.get('score', 0),
                'grade': verdict.get('grade', 'N/A'),
                'tests_passed': verdict.get('tests_passed', 0),
                'tests_failed': verdict.get('tests_failed', 0),
            }
            
            if verdict.get('score', 0) > comparison['best_score']:
                comparison['best_score'] = verdict['score']
                comparison['best_model'] = name
        
        return comparison
