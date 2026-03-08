"""
AEGIS AI - Generate HTML Audit Report
Trains models, audits them, and generates a visual HTML scorecard.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model_trainer import ModelTrainer
from core.bias_detector import BiasDetector
from core.report_generator import ReportGenerator


def generate_reports():
    print("🔧 Training models on recruitment dataset...")
    trainer = ModelTrainer()
    trainer.load_and_prepare_data(include_sensitive=True)
    trainer.train_all(include_debiased=False)

    detector = BiasDetector()
    generator = ReportGenerator()

    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports'), exist_ok=True)

    for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
        print(f"\n🔍 Auditing {model_name}...")
        model = trainer.models[model_name]

        audit = detector.audit_model(
            model=model,
            model_name=model_name,
            X_train=trainer.X_train,
            X_test=trainer.X_test,
            y_true=trainer.y_test,
            sensitive_features=trainer.sensitive_test,
            feature_names=trainer.feature_names,
        )

        verdict = audit['overall_verdict']
        print(f"   Score: {verdict['score']}% | Grade: {verdict['grade']} | {'PASS' if verdict['passed'] else 'FAIL'}")

        output_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'reports', f'audit_report_{model_name}.html'
        )
        generator.save_report(audit, output_path)
        print(f"   📄 Report saved: {output_path}")

    # Generate comparison
    comparison = detector.compare_models()
    print(f"\n🏆 Best model: {comparison.get('best_model', 'N/A')} (Score: {comparison.get('best_score', 0)}%)")
    print("\n✅ All reports generated in reports/ directory!")


if __name__ == "__main__":
    generate_reports()
