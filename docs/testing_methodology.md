# Testing Methodology — AEGIS AI

## Overview
AEGIS AI implements a comprehensive testing pipeline to validate bias detection accuracy, fairness metric correctness, and API reliability.

## Test Architecture

### 1. Unit Tests — Fairness Metrics (`test_fairness_metrics.py`)
**20 tests** covering all core fairness calculations:

| Test Category | Tests | Purpose |
|---|---|---|
| Demographic Parity | 5 | Validates equal selection rate computation across groups using the 4/5ths rule (threshold > 0.8) |
| Equal Opportunity | 4 | Verifies true positive rate calculations and max difference thresholds |
| Calibration | 3 | Tests Brier score computation for prediction confidence matching |
| Disparate Impact | 2 | Validates impact ratio calculations |
| Transparency Score | 3 | Ensures interpretability scoring for different model types |
| Privacy Check | 3 | Detects usage of sensitive features (gender, race, age) |
| Full Audit | 2 | End-to-end metric aggregation and pass rate percentage |

### 2. Integration Tests — Bias Detector (`test_bias_detector.py`)
**8 tests** validating the full audit pipeline:

- Audit result structure (dict with required keys)
- Performance metrics (accuracy present and valid)
- Fairness analysis (metrics computed correctly)
- Verdict generation (grade A-F, score 0-100)
- Recommendation engine (actionable suggestions generated)
- Model comparison (multi-model ranking by fairness score)

### 3. API Tests (`test_api.py`)
**9 tests** covering all FastAPI endpoints:

- `GET /health` — Health check returns 200 with status
- `GET /` — Root endpoint returns project metadata
- `POST /audit` — Full bias audit with JSON serialization
- `POST /compare` — Multi-model comparison
- `GET /report` — HTML report generation
- Error handling (invalid model returns 404)

## Running Tests

```bash
# Full suite
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=core --cov-report=html

# Specific module
python -m pytest tests/test_fairness_metrics.py -v
```

## Test Data
- **Synthetic dataset**: 2,000 candidates with engineered gender (+5 male bias), race (+4 white bias), and age (penalty for >45) biases
- **Local recruitment dataset**: 8,000+ real candidates from `dataset/data.csv` with Gender and HiringDecision attributes

## Validation Results
All **39/39 tests pass** consistently across runs (avg runtime: ~100s).
