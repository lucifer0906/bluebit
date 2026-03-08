"""
AEGIS AI - API Routes
"""

from fastapi import APIRouter, HTTPException
from backend.api.schemas import AuditRequest, CompareRequest
from core.model_trainer import ModelTrainer
from core.bias_detector import BiasDetector
from core.report_generator import ReportGenerator
from core.utils import sanitize_for_json
from typing import Dict, Any
import os

router = APIRouter()

ALLOWED_MODELS = {'logistic_regression', 'random_forest', 'xgboost'}


def _execute_audit(model, model_name: str, trainer: ModelTrainer) -> Dict[str, Any]:
    """Shared core logic to execute an audit given a loaded model and prepared trainer."""
    detector = BiasDetector()
    return detector.audit_model(
        model=model,
        model_name=model_name,
        X_train=trainer.X_train,
        X_test=trainer.X_test,
        y_true=trainer.y_test,
        sensitive_features=trainer.sensitive_test,
        feature_names=trainer.feature_names,
    )

def _run_audit(model_name: str, include_sensitive: bool = True) -> Dict[str, Any]:
    """Helper to load a model and run a single model audit."""
    if model_name not in ALLOWED_MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model_name}'. Allowed: {', '.join(sorted(ALLOWED_MODELS))}")
    trainer = ModelTrainer()
    trainer.load_and_prepare_data(include_sensitive=include_sensitive)
    
    # Try to load model from disk instead of training
    try:
        model = trainer.load_model(model_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found. Please train models first.")
        
    return _execute_audit(model, model_name, trainer)


@router.get("/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}


@router.post("/audit")
def run_audit(request: AuditRequest):
    """Run a bias audit on a specified model."""
    try:
        audit = _run_audit(request.model_name, request.include_sensitive_features)
        
        verdict = audit.get('overall_verdict', {})
        return sanitize_for_json({
            'model_name': audit['model_name'],
            'model_type': audit['model_type'],
            'accuracy': audit['performance']['accuracy'],
            'fairness_score': verdict.get('score', 0),
            'grade': verdict.get('grade', 'N/A'),
            'verdict': verdict.get('label', 'Unknown'),
            'tests_passed': verdict.get('tests_passed', 0),
            'tests_failed': verdict.get('tests_failed', 0),
            'total_tests': verdict.get('total_tests', 0),
            'metrics': audit['fairness']['metrics'],
            'recommendations': audit['recommendations'],
            'feature_importance': audit['explainability']['feature_importance'],
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
def compare_models(request: CompareRequest):
    """Compare multiple models for bias."""
    try:
        detector = BiasDetector()
        trainer = ModelTrainer()
        trainer.load_and_prepare_data(include_sensitive=request.include_sensitive_features)
        
        results = {}
        for model_name in request.model_names:
            try:
                model = trainer.load_model(model_name)
            except FileNotFoundError:
                raise HTTPException(status_code=400, detail=f"Model '{model_name}' not found. Cannot compare.")
            
            results[model_name] = _execute_audit(model, model_name, trainer)
            # Store in detector so compare_models() can access it
            detector.audit_results[model_name] = results[model_name]
        
        comparison = detector.compare_models()
        return sanitize_for_json(comparison)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/report/{model_name}")
def generate_report(model_name: str):
    """Generate an HTML audit report for a model."""
    try:
        audit = _run_audit(model_name)
        
        generator = ReportGenerator()
        report_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'tests', 'test_results'
        )
        os.makedirs(report_dir, exist_ok=True)
        output_path = os.path.join(report_dir, f'{model_name}_audit_report.html')
        
        generator.save_report(audit, output_path)
        
        return sanitize_for_json({
            'message': f'Report generated for {model_name}',
            'report_path': output_path,
            'summary': generator.generate_text_summary(audit)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
