"""
Unit Tests for BiasDetector
"""

import pytest
import numpy as np
import pandas as pd
from core.bias_detector import BiasDetector
from core.model_trainer import ModelTrainer


@pytest.fixture
def detector():
    return BiasDetector()


@pytest.fixture
def trained_model():
    """Train a quick model for testing."""
    trainer = ModelTrainer()
    trainer.load_and_prepare_data(include_sensitive=True)
    trainer.train_all()
    return trainer


class TestBiasDetector:
    def test_audit_returns_dict(self, detector, trained_model):
        model = trained_model.models['logistic_regression']
        result = detector.audit_model(
            model=model,
            model_name='logistic_regression',
            X_train=trained_model.X_train,
            X_test=trained_model.X_test,
            y_true=trained_model.y_test,
            sensitive_features=trained_model.sensitive_test,
            feature_names=trained_model.feature_names,
        )
        assert isinstance(result, dict)

    def test_audit_has_performance(self, detector, trained_model):
        model = trained_model.models['logistic_regression']
        result = detector.audit_model(
            model=model, model_name='logistic_regression',
            X_train=trained_model.X_train, X_test=trained_model.X_test,
            y_true=trained_model.y_test, sensitive_features=trained_model.sensitive_test,
            feature_names=trained_model.feature_names,
        )
        assert 'performance' in result
        assert 'accuracy' in result['performance']

    def test_audit_has_fairness(self, detector, trained_model):
        model = trained_model.models['logistic_regression']
        result = detector.audit_model(
            model=model, model_name='logistic_regression',
            X_train=trained_model.X_train, X_test=trained_model.X_test,
            y_true=trained_model.y_test, sensitive_features=trained_model.sensitive_test,
            feature_names=trained_model.feature_names,
        )
        assert 'fairness' in result

    def test_audit_has_verdict(self, detector, trained_model):
        model = trained_model.models['logistic_regression']
        result = detector.audit_model(
            model=model, model_name='logistic_regression',
            X_train=trained_model.X_train, X_test=trained_model.X_test,
            y_true=trained_model.y_test, sensitive_features=trained_model.sensitive_test,
            feature_names=trained_model.feature_names,
        )
        assert 'overall_verdict' in result
        verdict = result['overall_verdict']
        assert 'grade' in verdict
        assert 'score' in verdict
        assert 'passed' in verdict

    def test_verdict_grade_valid(self, detector, trained_model):
        model = trained_model.models['logistic_regression']
        result = detector.audit_model(
            model=model, model_name='logistic_regression',
            X_train=trained_model.X_train, X_test=trained_model.X_test,
            y_true=trained_model.y_test, sensitive_features=trained_model.sensitive_test,
            feature_names=trained_model.feature_names,
        )
        assert result['overall_verdict']['grade'] in ['A', 'B', 'C', 'D', 'F']

    def test_recommendations_exist(self, detector, trained_model):
        model = trained_model.models['logistic_regression']
        result = detector.audit_model(
            model=model, model_name='logistic_regression',
            X_train=trained_model.X_train, X_test=trained_model.X_test,
            y_true=trained_model.y_test, sensitive_features=trained_model.sensitive_test,
            feature_names=trained_model.feature_names,
        )
        assert 'recommendations' in result
        assert isinstance(result['recommendations'], list)

    def test_compare_models(self, detector, trained_model):
        models = {
            'lr': trained_model.models['logistic_regression'],
            'rf': trained_model.models['random_forest'],
        }
        results = detector.compare_models(
            models=models,
            X_train=trained_model.X_train,
            X_test=trained_model.X_test,
            y_true=trained_model.y_test,
            sensitive_features=trained_model.sensitive_test,
            feature_names=trained_model.feature_names,
        )
        assert len(results) == 2
        assert 'lr' in results
        assert 'rf' in results

    def test_score_between_0_and_100(self, detector, trained_model):
        model = trained_model.models['random_forest']
        result = detector.audit_model(
            model=model, model_name='random_forest',
            X_train=trained_model.X_train, X_test=trained_model.X_test,
            y_true=trained_model.y_test, sensitive_features=trained_model.sensitive_test,
            feature_names=trained_model.feature_names,
        )
        score = result['overall_verdict']['score']
        assert 0 <= score <= 100
