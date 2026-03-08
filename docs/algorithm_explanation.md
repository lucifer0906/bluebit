# Algorithm Explanation — AEGIS AI

## Bias Detection Pipeline

```
Dataset → Model Training → Prediction → Fairness Audit → Report
                                             │
                          ┌──────────────────┼──────────────────┐
                          │                  │                  │
                   Demographic         Equal              Adversarial
                    Parity          Opportunity           Debiasing
```

## Fairness Metrics

### 1. Demographic Parity
**Question:** Are selection rates equal across groups?

```
Ratio = min(selection_rate) / max(selection_rate)
PASS if ratio ≥ 0.8 (4/5ths rule)
```

A model passes if the least-selected group is hired at ≥80% the rate of the most-selected group.

### 2. Equal Opportunity
**Question:** Among qualified candidates, are true positive rates equal?

```
TPR(group) = True Positives / (True Positives + False Negatives)
PASS if max(TPR) - min(TPR) ≤ 0.1
```

### 3. Calibration
**Question:** When the model says "70% chance of hiring", does that match reality?

```
Brier Score = mean((predicted_probability - actual_outcome)²)
PASS if Brier Score < 0.25
```

### 4. Disparate Impact
**Question:** Does the model comply with the legal 4/5ths rule?

```
Impact Ratio = unfavorable_group_rate / favorable_group_rate
PASS if ratio ≥ 0.8
```

### 5. Transparency Score
**Question:** How interpretable is the model?

Scored 0-100 based on model type:
- Linear models (Logistic Regression): 80-90
- Tree-based (Random Forest, XGBoost): 60-70
- Neural Networks / Black-box: 30-50

## Explainability Engine (SHAP)

Uses **SHAP (SHapley Additive exPlanations)** to compute feature importance:

1. Train a background dataset summarizer (100 samples)
2. Compute SHAP values for each prediction
3. Rank features by mean absolute SHAP contribution
4. Flag sensitive features (gender, race) with >10% influence

## Adversarial Debiasing (AIF360)

### How It Works
Adversarial debiasing uses a **two-network architecture**:

```
Input Features → [Classifier Network] → Hiring Prediction
                         ↓
              [Adversary Network] → Sensitive Attribute Prediction
```

1. **Classifier** learns to predict hiring decisions
2. **Adversary** tries to predict gender/race from the classifier's internal representations
3. **Training**: The classifier is penalized when the adversary succeeds, forcing it to learn representations that **cannot** be used to predict sensitive attributes

### Key Parameters
| Parameter | Value | Purpose |
|---|---|---|
| `adversary_loss_weight` | 0.1 | Balance between accuracy and fairness |
| `num_epochs` | 50 | Training iterations |
| `batch_size` | 64 | Samples per training step |
| `debias` | True | Enable adversarial component |

### Implementation
- **Library**: AIF360 (IBM AI Fairness 360)
- **Backend**: TensorFlow v1 compatibility mode
- **Optimization**: RTX 4060 GPU + 20-thread CPU parallelism
- **Wrapper**: Custom scikit-learn compatible class (`AdversarialDebiaser`)

## Grading System

| Grade | Score | Interpretation |
|---|---|---|
| A | 90-100 | Excellent — Model is fair |
| B | 75-89 | Good — Minor improvements needed |
| C | 60-74 | Acceptable — Significant bias detected |
| D | 40-59 | Poor — Major fairness issues |
| F | 0-39 | Fail — Critical bias, do not deploy |
