# Testing Methodology — AEGIS AI

## Overview

AEGIS AI employs a multi-layered testing approach to validate the correctness of its bias detection algorithms, fairness metrics, and audit pipeline.

---

## 1. Unit Testing

### Fairness Metrics (`test_fairness_metrics.py`)
- **20+ test cases** covering all six core metrics
- Tests validate both fair and biased scenarios
- Each metric is tested for:
  - Correct output structure (keys, types)
  - Pass/fail accuracy with known inputs
  - Edge cases (perfect predictions, reversed predictions, equal groups)

### Bias Detector (`test_bias_detector.py`)
- End-to-end audit pipeline validation
- Verifies presence of all required output fields: `performance`, `fairness`, `overall_verdict`, `recommendations`
- Validates grade computation (A–F scale) and score range (0–100)
- Multi-model comparison tests

### API Tests (`test_api.py`)
- FastAPI endpoint testing with `TestClient`
- Health check, audit, compare, and report endpoints
- Error handling for invalid model names

---

## 2. Metrics Validation Approach

### Demographic Parity
- **Method**: Compare selection rates across protected groups
- **Threshold**: 4/5ths rule (ratio ≥ 0.8)
- **Test Strategy**: Create known biased data (Male 90% vs Female 30%) and verify failure; create balanced data and verify pass

### Equal Opportunity
- **Method**: Compare True Positive Rates across groups
- **Threshold**: Max TPR difference ≤ 0.1
- **Test Strategy**: Perfect predictions should pass; skewed predictions should fail

### Calibration
- **Method**: Brier score computation
- **Threshold**: Score ≤ 0.25
- **Test Strategy**: Well-calibrated probabilities should pass; reversed probabilities should fail

### Disparate Impact
- **Method**: Ratio of selection rates between groups
- **Threshold**: Ratio ≥ 0.8 (same as demographic parity, different computation)
- **Test Strategy**: Equal outcomes should always pass

### Transparency Score
- **Method**: Model-type-based interpretability rating
- **Test Strategy**: Logistic regression should score higher than ensemble/boosting models

### Privacy Check
- **Method**: Feature name analysis for sensitive attributes
- **Test Strategy**: Features like 'gender', 'race', 'age' should be flagged; technical features should pass

---

## 3. Integration Testing

- Full pipeline test: data generation → model training → audit → report
- API integration: POST audit request → verify response structure
- Frontend-backend communication validation

---

## 4. Test Data Strategy

| Dataset | Purpose | Size |
|---------|---------|------|
| Synthetic hiring data | Primary test dataset | 2000 rows |
| Known biased fixtures | Metric validation | 200 rows |
| Edge case arrays | Boundary testing | 4-8 rows |

### Bias Injection
The synthetic dataset intentionally injects:
- **Gender bias**: +5 skill for Male, -3 for Female
- **Race bias**: +4 for White, -3 for Black, -2 for Hispanic
- **Age bias**: -4 for candidates over 45

This ensures the audit system correctly detects known biases.

---

## 5. Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=core --cov-report=html

# Run specific test file
pytest tests/test_fairness_metrics.py -v

# Run specific test class
pytest tests/test_fairness_metrics.py::TestDemographicParity -v
```

---

## 6. Continuous Validation

- Tests are designed to run independently (no shared state)
- Fixtures provide fresh data for each test
- Random seeds (42) ensure reproducible results
- All tests should complete in under 60 seconds
