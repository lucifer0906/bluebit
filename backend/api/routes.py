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


def _run_audit(model_name: str, include_sensitive: bool = True) -> Dict[str, Any]:
    """Helper to run a single model audit."""
    trainer = ModelTrainer()
    trainer.load_and_prepare_data(include_sensitive=include_sensitive)
    trainer.train_all()
    
    model = trainer.models.get(model_name)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    detector = BiasDetector()
    y_pred, y_prob = trainer.get_predictions(model_name=model_name)
    
    audit = detector.audit_model(
        model=model,
        model_name=model_name,
        X_train=trainer.X_train,
        X_test=trainer.X_test,
        y_true=trainer.y_test,
        sensitive_features=trainer.sensitive_test,
        feature_names=trainer.feature_names,
    )
    
    return audit


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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
def compare_models(request: CompareRequest):
    """Compare multiple models for bias."""
    try:
        detector = BiasDetector()
        trainer = ModelTrainer()
        trainer.load_and_prepare_data(include_sensitive=request.include_sensitive_features)
        trainer.train_all()
        
        results = {}
        for model_name in request.model_names:
            model = trainer.models.get(model_name)
            if model is None:
                continue
            
            audit = detector.audit_model(
                model=model,
                model_name=model_name,
                X_train=trainer.X_train,
                X_test=trainer.X_test,
                y_true=trainer.y_test,
                sensitive_features=trainer.sensitive_test,
                feature_names=trainer.feature_names,
            )
            results[model_name] = audit
        
        comparison = detector.compare_models()
        return sanitize_for_json(comparison)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/report/{model_name}")
def generate_report(model_name: str):
    """Generate an HTML audit report for a model."""
    try:
        audit = _run_audit(model_name)
        
        generator = ReportGenerator()
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'tests', 'test_results', f'{model_name}_audit_report.html'
        )
        generator.save_report(audit, output_path)
        
        return sanitize_for_json({
            'message': f'Report generated for {model_name}',
            'report_path': output_path,
            'summary': generator.generate_text_summary(audit)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
