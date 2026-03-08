"""
AEGIS AI - Fairness Metrics Calculator
Implements ethical testing metrics for AI model auditing.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, brier_score_loss
from typing import Dict, List, Tuple, Any


class FairnessMetrics:
    """Calculate fairness metrics for ML model predictions."""
    
    def __init__(self, y_true: np.ndarray = None, y_pred: np.ndarray = None, 
                 y_prob: np.ndarray = None, sensitive_features: pd.DataFrame = None):
        """
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            sensitive_features: DataFrame with sensitive attribute columns
        """
        if y_true is not None:
            self.y_true = np.array(y_true)
        if y_pred is not None:
            self.y_pred = np.array(y_pred)
        if y_prob is not None:
            self.y_prob = np.array(y_prob)
        if sensitive_features is not None:
            self.sensitive_features = sensitive_features
    
    def demographic_parity(self, y_pred=None, groups=None, attribute: str = None) -> Dict[str, Any]:
        """
        Demographic Parity: Equal selection rates across groups.
        
        Can be called two ways:
          1. demographic_parity(attribute='gender')  — uses self.y_pred & self.sensitive_features
          2. demographic_parity(y_pred, groups)       — standalone call
        """
        if y_pred is not None and groups is not None:
            # Standalone mode
            y_pred = np.array(y_pred)
            groups_series = pd.Series(groups)
            unique_groups = groups_series.unique()
            selection_rates = {}
            for group in unique_groups:
                mask = groups_series == group
                rate = y_pred[mask].mean()
                selection_rates[group] = round(float(rate), 4)
        elif attribute is not None:
            # Instance mode
            unique_groups = self.sensitive_features[attribute].unique()
            selection_rates = {}
            for group in unique_groups:
                mask = self.sensitive_features[attribute] == group
                rate = self.y_pred[mask].mean()
                selection_rates[group] = round(float(rate), 4)
        else:
            raise ValueError("Provide either (y_pred, groups) or attribute=...")
        
        rates = list(selection_rates.values())
        min_rate = min(rates)
        max_rate = max(rates)
        
        # Disparate impact ratio (4/5ths rule)
        ratio = min_rate / max_rate if max_rate > 0 else 0
        
        return {
            'metric': 'Demographic Parity',
            'attribute': attribute,
            'selection_rates': selection_rates,
            'min_rate': min_rate,
            'max_rate': max_rate,
            'parity_ratio': round(ratio, 4),
            'threshold': 0.8,
            'passed': ratio >= 0.8,
            'description': 'Measures whether selection rates are equal across demographic groups. '
                          'The 4/5ths rule requires the ratio to be >= 0.8.'
        }
    
    def equal_opportunity(self, y_true=None, y_pred=None, groups=None, 
                          attribute: str = None) -> Dict[str, Any]:
        """
        Equal Opportunity: Equal true positive rates across groups.
        
        Can be called two ways:
          1. equal_opportunity(attribute='gender')
          2. equal_opportunity(y_true, y_pred, groups)
        """
        if y_true is not None and y_pred is not None and groups is not None:
            y_true_arr = np.array(y_true)
            y_pred_arr = np.array(y_pred)
            groups_series = pd.Series(groups)
            unique_groups = groups_series.unique()
        elif attribute is not None:
            y_true_arr = self.y_true
            y_pred_arr = self.y_pred
            groups_series = self.sensitive_features[attribute]
            unique_groups = groups_series.unique()
        else:
            raise ValueError("Provide either (y_true, y_pred, groups) or attribute=...")
        
        tpr_rates = {}
        for group in unique_groups:
            mask = groups_series == group
            group_true = y_true_arr[mask]
            group_pred = y_pred_arr[mask]
            
            positives = group_true == 1
            if positives.sum() > 0:
                tpr = (group_pred[positives] == 1).mean()
            else:
                tpr = 0.0
            tpr_rates[group] = round(float(tpr), 4)
        
        rates = list(tpr_rates.values())
        max_diff = max(rates) - min(rates)
        
        return {
            'metric': 'Equal Opportunity',
            'attribute': attribute,
            'true_positive_rates': tpr_rates,
            'max_difference': round(max_diff, 4),
            'threshold': 0.1,
            'passed': max_diff <= 0.1,
            'description': 'Measures whether qualified candidates from all groups have '
                          'equal chances of being selected. Difference should be <= 0.1.'
        }
    
    def calibration(self, y_true=None, y_prob=None, n_bins: int = 10) -> Dict[str, Any]:
        """
        Calibration: Predicted probabilities match actual outcomes.
        """
        y_t = np.array(y_true) if y_true is not None else self.y_true
        y_p = np.array(y_prob) if y_prob is not None else self.y_prob
        
        brier = brier_score_loss(y_t, y_p)
        
        # Binned calibration
        bins = np.linspace(0, 1, n_bins + 1)
        bin_data = []
        
        for i in range(n_bins):
            mask = (y_p >= bins[i]) & (y_p < bins[i+1])
            if mask.sum() > 0:
                predicted_avg = y_p[mask].mean()
                actual_avg = y_t[mask].mean()
                bin_data.append({
                    'bin': f'{bins[i]:.1f}-{bins[i+1]:.1f}',
                    'count': int(mask.sum()),
                    'predicted_avg': round(predicted_avg, 4),
                    'actual_avg': round(actual_avg, 4),
                    'gap': round(abs(predicted_avg - actual_avg), 4)
                })
        
        return {
            'metric': 'Calibration',
            'brier_score': round(brier, 4),
            'threshold': 0.25,
            'passed': brier < 0.25,
            'bin_data': bin_data,
            'description': 'Measures whether predicted probabilities match actual outcomes. '
                          'Brier score should be < 0.25 for good calibration.'
        }
    
    def disparate_impact(self, y_pred=None, groups=None, attribute: str = None) -> Dict[str, Any]:
        """
        Disparate Impact: 4/5ths rule.
        """
        dp = self.demographic_parity(y_pred=y_pred, groups=groups, attribute=attribute)
        return {
            'metric': 'Disparate Impact',
            'attribute': attribute,
            'impact_ratio': dp['parity_ratio'],
            'threshold': 0.8,
            'passed': dp['parity_ratio'] >= 0.8,
            'selection_rates': dp['selection_rates'],
            'description': 'The 4/5ths rule: the selection rate for any group should be '
                          'at least 80% of the highest group\'s rate.'
        }
    
    def transparency_score(self, model, feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Transparency/Interpretability Score.
        Rates the model based on interpretability factors.
        model can be a model object or a string model type name.
        """
        if isinstance(model, str):
            model_type = model
        else:
            model_type = type(model).__name__
        
        # Interpretability scoring (keys are both class names and snake_case aliases)
        interpretable_models = {
            'LogisticRegression': 90,
            'logistic_regression': 90,
            'DecisionTreeClassifier': 85,
            'decision_tree': 85,
            'LinearSVC': 80,
            'RandomForestClassifier': 60,
            'random_forest': 60,
            'GradientBoostingClassifier': 50,
            'XGBClassifier': 45,
            'xgboost': 45,
            'MLPClassifier': 30,
            'SVC': 25,
        }
        
        base_score = interpretable_models.get(model_type, 40)
        
        # Bonus for fewer features (more interpretable)
        n_features = len(feature_names) if feature_names else 0
        feature_penalty = max(0, (n_features - 10) * 2)
        
        score = max(0, min(100, base_score - feature_penalty))
        
        return {
            'metric': 'Transparency Score',
            'model_type': model_type,
            'score': score,
            'threshold': 60,
            'passed': score >= 60,
            'n_features': n_features,
            'description': f'Interpretability score for {model_type}. '
                          f'Score >= 60 indicates acceptable transparency.'
        }
    
    def privacy_check(self, feature_names: List[str]) -> Dict[str, Any]:
        """
        Privacy Preservation: Check if sensitive attributes are used as features.
        """
        sensitive_attrs = {'gender', 'race', 'ethnicity', 'religion', 'sexual_orientation',
                          'disability', 'marital_status', 'pregnancy', 'national_origin',
                          'age', 'sex', 'skin_color', 'nationality'}
        
        used_sensitive = [f for f in feature_names if f.lower() in sensitive_attrs]
        
        return {
            'metric': 'Privacy Preservation',
            'sensitive_features_used': used_sensitive,
            'total_features': len(feature_names),
            'passed': len(used_sensitive) == 0,
            'description': 'Checks whether sensitive/protected attributes are directly '
                          'used as model features. No sensitive attributes should be used.'
        }
    
    def run_full_audit(self, model=None, feature_names: List[str] = None,
                       sensitive_attributes: List[str] = None,
                       y_true=None, y_pred=None, y_prob=None,
                       sensitive_features=None, model_type=None) -> Dict[str, Any]:
        """
        Run a complete fairness audit across all metrics.
        
        Supports both instance mode (data set in constructor) and
        standalone mode (data passed as arguments).
        """
        # Allow standalone call with all data passed in
        if y_true is not None:
            self.y_true = np.array(y_true)
        if y_pred is not None:
            self.y_pred = np.array(y_pred)
        if y_prob is not None:
            self.y_prob = np.array(y_prob)
        if sensitive_features is not None:
            self.sensitive_features = sensitive_features
        
        if sensitive_attributes is None:
            sensitive_attributes = list(self.sensitive_features.columns)
        
        results = {
            'summary': {
                'total_samples': len(self.y_true),
                'positive_rate': round(self.y_true.mean(), 4),
                'predicted_positive_rate': round(self.y_pred.mean(), 4),
            },
            'metrics': {},
            'overall_passed': True,
            'pass_count': 0,
            'fail_count': 0,
            'total_tests': 0,
        }
        
        # Run demographic parity for each sensitive attribute
        for attr in sensitive_attributes:
            key = f'demographic_parity_{attr}'
            result = self.demographic_parity(attribute=attr)
            results['metrics'][key] = result
            results['total_tests'] += 1
            if result['passed']:
                results['pass_count'] += 1
            else:
                results['fail_count'] += 1
                results['overall_passed'] = False
        
        # Run equal opportunity for each sensitive attribute
        for attr in sensitive_attributes:
            key = f'equal_opportunity_{attr}'
            result = self.equal_opportunity(attribute=attr)
            results['metrics'][key] = result
            results['total_tests'] += 1
            if result['passed']:
                results['pass_count'] += 1
            else:
                results['fail_count'] += 1
                results['overall_passed'] = False
        
        # Run calibration
        cal_result = self.calibration()
        results['metrics']['calibration'] = cal_result
        results['total_tests'] += 1
        if cal_result['passed']:
            results['pass_count'] += 1
        else:
            results['fail_count'] += 1
            results['overall_passed'] = False
        
        # Run disparate impact
        for attr in sensitive_attributes:
            key = f'disparate_impact_{attr}'
            result = self.disparate_impact(attribute=attr)
            results['metrics'][key] = result
            results['total_tests'] += 1
            if result['passed']:
                results['pass_count'] += 1
            else:
                results['fail_count'] += 1
                results['overall_passed'] = False
        
        # Run transparency check if model provided
        if model is not None and feature_names is not None:
            trans_result = self.transparency_score(model, feature_names)
            results['metrics']['transparency'] = trans_result
            results['total_tests'] += 1
            if trans_result['passed']:
                results['pass_count'] += 1
            else:
                results['fail_count'] += 1
                results['overall_passed'] = False
        
        # Run privacy check if feature names provided
        if feature_names is not None:
            priv_result = self.privacy_check(feature_names)
            results['metrics']['privacy'] = priv_result
            results['total_tests'] += 1
            if priv_result['passed']:
                results['pass_count'] += 1
            else:
                results['fail_count'] += 1
                results['overall_passed'] = False
        
        # Calculate overall fairness score
        results['fairness_score'] = round(
            (results['pass_count'] / results['total_tests']) * 100, 1
        ) if results['total_tests'] > 0 else 0
        
        results['pass_rate'] = results['fairness_score']
        
        return results
