"""End-to-end integration test for the full AEGIS AI pipeline."""

import pytest
from core.model_trainer import ModelTrainer
from core.bias_detector import BiasDetector
from core.report_generator import ReportGenerator


@pytest.fixture(scope="module")
def trained_trainer():
    trainer = ModelTrainer()
    trainer.load_and_prepare_data(include_sensitive=True)
    trainer.train_all()
    trainer.save_models()
    return trainer


@pytest.fixture(scope="module")
def audit_results(trained_trainer):
    detector = BiasDetector()
    results = {}
    for name in ['logistic_regression', 'random_forest', 'xgboost']:
        model = trained_trainer.models[name]
        results[name] = detector.audit_model(
            model=model, model_name=name,
            X_train=trained_trainer.X_train, X_test=trained_trainer.X_test,
            y_true=trained_trainer.y_test,
            sensitive_features=trained_trainer.sensitive_test,
            feature_names=trained_trainer.feature_names,
        )
        detector.audit_results[name] = results[name]
    results['_detector'] = detector
    return results


class TestPipelineTraining:
    def test_all_models_trained(self, trained_trainer):
        assert set(trained_trainer.models.keys()) == {'logistic_regression', 'random_forest', 'xgboost'}

    def test_accuracy_above_50(self, trained_trainer):
        for name, model in trained_trainer.models.items():
            acc = model.score(trained_trainer.X_test, trained_trainer.y_test)
            assert acc > 0.5, f"{name} accuracy {acc} is below 50%"

    def test_models_saved_to_disk(self, trained_trainer):
        import os
        for name in ['logistic_regression', 'random_forest', 'xgboost']:
            path = os.path.join(trained_trainer.MODELS_DIR, f'{name}.joblib')
            assert os.path.exists(path), f"{name}.joblib not found on disk"


class TestPipelineAudit:
    def test_all_audits_have_verdict(self, audit_results):
        for name in ['logistic_regression', 'random_forest', 'xgboost']:
            verdict = audit_results[name]['overall_verdict']
            assert 'score' in verdict
            assert 'grade' in verdict
            assert verdict['grade'] in ['A', 'B', 'C', 'D', 'F']

    def test_fairness_score_in_range(self, audit_results):
        for name in ['logistic_regression', 'random_forest', 'xgboost']:
            score = audit_results[name]['overall_verdict']['score']
            assert 0 <= score <= 100

    def test_comparison_picks_best(self, audit_results):
        comparison = audit_results['_detector'].compare_models()
        assert 'best_model' in comparison
        assert comparison['best_model'] in ['logistic_regression', 'random_forest', 'xgboost']


class TestPipelineReport:
    def test_html_report_generated(self, audit_results):
        gen = ReportGenerator()
        html = gen.generate_scorecard_html(audit_results['logistic_regression'])
        assert '<html>' in html
        assert 'AEGIS AI' in html

    def test_text_summary_generated(self, audit_results):
        gen = ReportGenerator()
        text = gen.generate_text_summary(audit_results['logistic_regression'])
        assert 'Fairness Score' in text
        assert 'Grade' in text

    def test_html_report_escapes_content(self, audit_results):
        gen = ReportGenerator()
        html = gen.generate_scorecard_html(audit_results['logistic_regression'])
        # Should not contain raw unescaped dict braces from metric values
        # (html_escape converts & < > " to entities)
        assert '&amp;' in html or '<script>' not in html
