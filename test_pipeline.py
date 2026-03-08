"""Quick smoke test for the full AEGIS AI pipeline."""

from core.model_trainer import ModelTrainer
from core.bias_detector import BiasDetector
from core.report_generator import ReportGenerator

print("=" * 60)
print("  AEGIS AI — Full Pipeline Smoke Test")
print("=" * 60)

# 1. Train models
print("\n1. Training models...")
trainer = ModelTrainer()
trainer.load_and_prepare_data(include_sensitive=True)
trainer.train_all()
trainer.save_models()

for name, model in trainer.models.items():
    acc = model.score(trainer.X_test, trainer.y_test)
    print(f"   {name}: accuracy = {acc:.4f}")

# 2. Run bias audit on each model
print("\n2. Running bias audits...")
detector = BiasDetector()

for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
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
    
    v = audit['overall_verdict']
    print(f"   {model_name}: Grade={v['grade']} | Score={v['score']}% | "
          f"Passed={v['passed']} | Tests={v['tests_passed']}/{v['total_tests']}")

# 3. Compare models
print("\n3. Model comparison...")
comparison = detector.compare_models()
if 'best_model' in comparison:
    print(f"   Best model: {comparison['best_model']} (score: {comparison['best_score']}%)")

# 4. Generate report
print("\n4. Generating HTML report...")
report_gen = ReportGenerator()
audit = detector.audit_results['logistic_regression']
report_path = report_gen.save_report(audit, 'tests/test_results/logistic_regression_audit.html')

text_summary = report_gen.generate_text_summary(audit)
print(text_summary)

print("\n✅ All pipeline stages completed successfully!")
