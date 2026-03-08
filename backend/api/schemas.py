"""
AEGIS AI - API Schemas (Pydantic Models)
"""

from pydantic import BaseModel
from typing import Dict, List, Any, Optional


class AuditRequest(BaseModel):
    model_name: str = "logistic_regression"
    include_sensitive_features: bool = True
    sensitive_attributes: List[str] = ["gender", "race"]


class AuditResponse(BaseModel):
    model_name: str
    model_type: str
    accuracy: float
    fairness_score: float
    grade: str
    verdict: str
    tests_passed: int
    tests_failed: int
    total_tests: int
    metrics: Dict[str, Any]
    recommendations: List[Dict[str, str]]
    feature_importance: Dict[str, float]


class CompareRequest(BaseModel):
    model_names: List[str] = ["logistic_regression", "random_forest", "xgboost"]
    include_sensitive_features: bool = True


class HealthResponse(BaseModel):
    status: str
    version: str
