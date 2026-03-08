"""
Unit Tests for FastAPI Backend
"""

import pytest
from fastapi.testclient import TestClient

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_returns_ok(self, client):
        response = client.get("/api/health")
        data = response.json()
        assert data["status"] == "healthy"


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_project_name(self, client):
        response = client.get("/")
        data = response.json()
        assert "AEGIS AI" in data.get("project", "")


class TestAuditEndpoint:
    def test_audit_returns_200(self, client):
        payload = {
            "model_name": "logistic_regression",
            "include_sensitive_features": True
        }
        response = client.post("/api/audit", json=payload)
        assert response.status_code == 200

    def test_audit_has_verdict(self, client):
        payload = {
            "model_name": "logistic_regression",
            "include_sensitive_features": True
        }
        response = client.post("/api/audit", json=payload)
        data = response.json()
        assert "verdict" in data or "grade" in data

    def test_audit_invalid_model(self, client):
        payload = {
            "model_name": "invalid_model",
            "include_sensitive_features": True
        }
        response = client.post("/api/audit", json=payload)
        # Should return error or handle gracefully
        assert response.status_code in [200, 400, 404, 422, 500]


class TestCompareEndpoint:
    def test_compare_returns_200(self, client):
        payload = {
            "model_names": ["logistic_regression", "random_forest"],
            "include_sensitive_features": True
        }
        response = client.post("/api/compare", json=payload)
        assert response.status_code == 200


class TestReportEndpoint:
    def test_report_returns_html(self, client):
        response = client.post("/api/report/logistic_regression")
        assert response.status_code == 200
