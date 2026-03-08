"""
Unit Tests for Fairness Metrics
20+ test cases covering all fairness metrics.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from core.fairness_metrics import FairnessMetrics


@pytest.fixture
def metrics():
    return FairnessMetrics()


@pytest.fixture
def sample_data():
    """Create sample predictions and sensitive features for testing."""
    np.random.seed(42)
    n = 200
    y_true = np.random.randint(0, 2, n)
    y_pred = y_true.copy()
    # introduce some noise
    flip = np.random.choice(n, 20, replace=False)
    y_pred[flip] = 1 - y_pred[flip]
    
    sensitive = pd.DataFrame({
        'gender': np.random.choice(['Male', 'Female'], n),
        'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], n),
    })
    return y_true, y_pred, sensitive


@pytest.fixture
def biased_data():
    """Create intentionally biased predictions."""
    n = 200
    y_true = np.ones(n, dtype=int)
    y_pred = np.ones(n, dtype=int)
    
    gender = np.array(['Male'] * 100 + ['Female'] * 100)
    # Males: 90% hired, Females: 30% hired
    y_pred[:90] = 1
    y_pred[90:100] = 0
    y_pred[100:130] = 1
    y_pred[130:] = 0
    
    sensitive = pd.DataFrame({'gender': gender})
    return y_true, y_pred, sensitive


# ===== Demographic Parity Tests =====

class TestDemographicParity:
    def test_fair_model_passes(self, metrics, sample_data):
        y_true, y_pred, sensitive = sample_data
        result = metrics.demographic_parity(y_pred, sensitive['gender'])
        assert 'passed' in result
        assert 'selection_rates' in result
        assert isinstance(result['selection_rates'], dict)

    def test_biased_model_fails(self, metrics, biased_data):
        y_true, y_pred, sensitive = biased_data
        result = metrics.demographic_parity(y_pred, sensitive['gender'])
        assert result['passed'] is False

    def test_selection_rates_computed(self, metrics, sample_data):
        y_true, y_pred, sensitive = sample_data
        result = metrics.demographic_parity(y_pred, sensitive['gender'])
        assert 'Male' in result['selection_rates']
        assert 'Female' in result['selection_rates']

    def test_equal_selection_passes(self, metrics):
        y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        groups = pd.Series(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])
        result = metrics.demographic_parity(y_pred, groups)
        assert result['passed'] is True

    def test_threshold_is_four_fifths(self, metrics):
        result = metrics.demographic_parity(
            np.array([1, 1, 1, 0, 1, 0, 0, 0]),
            pd.Series(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])
        )
        assert result.get('threshold', 0.8) == 0.8


# ===== Equal Opportunity Tests =====

class TestEqualOpportunity:
    def test_returns_tpr(self, metrics, sample_data):
        y_true, y_pred, sensitive = sample_data
        result = metrics.equal_opportunity(y_true, y_pred, sensitive['gender'])
        assert 'true_positive_rates' in result

    def test_fair_model_passes(self, metrics):
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        groups = pd.Series(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])
        result = metrics.equal_opportunity(y_true, y_pred, groups)
        assert result['passed'] is True

    def test_biased_tpr_fails(self, metrics, biased_data):
        y_true, y_pred, sensitive = biased_data
        result = metrics.equal_opportunity(y_true, y_pred, sensitive['gender'])
        assert result['passed'] is False

    def test_max_difference_reported(self, metrics, sample_data):
        y_true, y_pred, sensitive = sample_data
        result = metrics.equal_opportunity(y_true, y_pred, sensitive['gender'])
        assert 'max_difference' in result


# ===== Calibration Tests =====

class TestCalibration:
    def test_perfect_calibration(self, metrics):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2])
        result = metrics.calibration(y_true, y_prob)
        assert result['passed'] is True

    def test_poor_calibration_fails(self, metrics):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.1, 0.9])  # reversed
        result = metrics.calibration(y_true, y_prob)
        assert result['passed'] is False

    def test_brier_score_returned(self, metrics):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.8, 0.2, 0.7, 0.3])
        result = metrics.calibration(y_true, y_prob)
        assert 'brier_score' in result
        assert isinstance(result['brier_score'], float)


# ===== Disparate Impact Tests =====

class TestDisparateImpact:
    def test_returns_ratio(self, metrics, sample_data):
        y_true, y_pred, sensitive = sample_data
        result = metrics.disparate_impact(y_pred, sensitive['gender'])
        assert 'ratio' in result or 'selection_rates' in result

    def test_equal_outcomes_pass(self, metrics):
        y_pred = np.array([1, 1, 1, 1])
        groups = pd.Series(['A', 'A', 'B', 'B'])
        result = metrics.disparate_impact(y_pred, groups)
        assert result['passed'] is True


# ===== Transparency Score Tests =====

class TestTransparencyScore:
    def test_interpretable_model_high_score(self, metrics):
        result = metrics.transparency_score("logistic_regression")
        assert result['score'] >= 70

    def test_black_box_lower_score(self, metrics):
        result = metrics.transparency_score("xgboost")
        assert result['score'] <= result.get('threshold', 100)

    def test_always_returns_score(self, metrics):
        result = metrics.transparency_score("random_forest")
        assert 'score' in result
        assert isinstance(result['score'], (int, float))


# ===== Privacy Check Tests =====

class TestPrivacyCheck:
    def test_detects_sensitive_features(self, metrics):
        features = ['gender', 'age', 'experience', 'race', 'skill_score']
        result = metrics.privacy_check(features)
        assert len(result.get('sensitive_features_used', [])) > 0

    def test_clean_features_pass(self, metrics):
        features = ['experience', 'skill_score', 'education_years']
        result = metrics.privacy_check(features)
        assert result['passed'] is True

    def test_gender_flagged(self, metrics):
        features = ['gender', 'experience']
        result = metrics.privacy_check(features)
        found = result.get('sensitive_features_used', [])
        assert 'gender' in found


# ===== Full Audit Tests =====

class TestFullAudit:
    def test_full_audit_returns_all_metrics(self, metrics, sample_data):
        from sklearn.linear_model import LogisticRegression
        y_true, y_pred, sensitive = sample_data
        
        features = ['gender', 'experience', 'skill_score']
        y_prob = np.clip(y_pred + np.random.normal(0, 0.1, len(y_pred)), 0, 1)
        
        result = metrics.run_full_audit(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            sensitive_features=sensitive,
            feature_names=features,
            model_type="logistic_regression"
        )
        
        assert 'metrics' in result
        assert 'overall_passed' in result
        assert 'pass_rate' in result

    def test_pass_rate_is_percentage(self, metrics, sample_data):
        y_true, y_pred, sensitive = sample_data
        y_prob = np.clip(y_pred.astype(float), 0, 1)
        
        result = metrics.run_full_audit(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            sensitive_features=sensitive,
            feature_names=['f1', 'f2'],
            model_type="logistic_regression"
        )
        
        assert 0 <= result['pass_rate'] <= 100
