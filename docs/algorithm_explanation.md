# Algorithm Explanation — AEGIS AI

## Ethical AI Auditing Framework for Hiring Systems

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    AEGIS AI Pipeline                 │
│                                                     │
│  Dataset → Model Training → Bias Audit → Report     │
│     ↓           ↓              ↓           ↓        │
│  Synthetic   LR/RF/XGB    6 Metrics    Scorecard    │
│  Biased      Models       + SHAP       HTML/Text    │
│  Data                     Analysis                  │
└─────────────────────────────────────────────────────┘
```

---

## 2. Fairness Metrics — Detection Algorithms

### 2.1 Demographic Parity (Statistical Parity)

**Question**: Are positive outcomes distributed equally across groups?

**Algorithm**:
$$\text{Selection Rate}_g = \frac{|\{i : \hat{y}_i = 1, g_i = g\}|}{|\{i : g_i = g\}|}$$

$$\text{Parity Ratio} = \frac{\min_g \text{Selection Rate}_g}{\max_g \text{Selection Rate}_g}$$

**Threshold**: Ratio ≥ 0.8 (the "four-fifths rule" from EEOC guidelines)

**Interpretation**: If male candidates are hired at 60% and female at 45%, the ratio is 45/60 = 0.75, which FAILS.

---

### 2.2 Equal Opportunity

**Question**: Among qualified candidates, are positive predictions equally likely?

**Algorithm**:
$$\text{TPR}_g = \frac{\text{True Positives}_g}{\text{True Positives}_g + \text{False Negatives}_g}$$

$$\text{Max Difference} = \max_g \text{TPR}_g - \min_g \text{TPR}_g$$

**Threshold**: Max difference ≤ 0.1

**Interpretation**: If the model correctly identifies 90% of qualified males but only 70% of qualified females, the difference is 0.2, which FAILS.

---

### 2.3 Calibration (Brier Score)

**Question**: Are the model's probability estimates reliable?

**Algorithm**:
$$\text{Brier Score} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i)^2$$

where $p_i$ is the predicted probability and $y_i$ is the true outcome.

**Threshold**: Brier Score ≤ 0.25

**Interpretation**: A perfectly calibrated model has a Brier score of 0. A score above 0.25 indicates unreliable probability estimates.

---

### 2.4 Disparate Impact

**Question**: Does the model disproportionately affect a protected group?

**Algorithm**: Same as Demographic Parity but applied across all protected attributes simultaneously:

$$\text{DI} = \frac{P(\hat{y}=1 | g = \text{unprivileged})}{P(\hat{y}=1 | g = \text{privileged})}$$

**Threshold**: DI ≥ 0.8

**Legal Basis**: Title VII of the Civil Rights Act — the 80% rule.

---

### 2.5 Transparency Score

**Question**: How interpretable is the model?

**Algorithm** (heuristic):
| Model Type | Score |
|---|---|
| Logistic Regression | 90/100 |
| Decision Tree | 85/100 |
| Random Forest | 65/100 |
| XGBoost | 55/100 |
| Neural Network | 30/100 |

**Threshold**: Score ≥ 50 (configurable)

**Rationale**: Simpler models allow stakeholders to understand and contest decisions.

---

### 2.6 Privacy Check

**Question**: Does the model use sensitive attributes directly?

**Algorithm**: Pattern matching on feature names against known sensitive categories:
- Gender indicators: `gender`, `sex`, `male`, `female`
- Race indicators: `race`, `ethnicity`, `color`
- Age indicators: `age`, `birth`, `dob`

**Result**: List of sensitive features found; PASS if none detected.

---

## 3. Explainability — SHAP Analysis

### How SHAP Works

SHAP (SHapley Additive exPlanations) is based on cooperative game theory:

$$\phi_j = \sum_{S \subseteq N \setminus \{j\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [f(S \cup \{j\}) - f(S)]$$

where:
- $\phi_j$ is the SHAP value for feature $j$
- $N$ is the set of all features
- $f(S)$ is the model prediction using feature subset $S$

### What We Compute

1. **Global Feature Importance**: Mean |SHAP value| per feature — shows which features matter most overall
2. **Individual Explanations**: Per-candidate SHAP values — explains why each decision was made
3. **Bias in Explanations**: Checks if sensitive features have > 10% importance

---

## 4. Scoring and Grading

### Overall Fairness Score

$$\text{Score} = \frac{\text{Tests Passed}}{\text{Total Tests}} \times 100$$

### Grade Mapping

| Grade | Score Range | Interpretation |
|-------|-----------|----------------|
| A | ≥ 90% | Excellent — minimal bias detected |
| B | ≥ 75% | Good — minor concerns |
| C | ≥ 60% | Fair — needs attention |
| D | ≥ 40% | Poor — significant bias |
| F | < 40% | Failing — critical bias issues |

### Pass/Fail Decision
A model **passes** the audit if its score ≥ 60% (Grade C or above).

---

## 5. Recommendation Engine

When a metric fails, the system generates severity-rated recommendations:

| Severity | Trigger | Example |
|----------|---------|---------|
| CRITICAL | Demographic parity fails | "Selection rates differ by >20% across genders — retrain with fairness constraints" |
| HIGH | Equal opportunity fails | "True positive rate gap >10% — audit training data distribution" |
| MEDIUM | Privacy check fails | "Sensitive features in model — consider removing or using proxy-free alternatives" |
| LOW | Transparency below threshold | "Consider using a more interpretable model for regulatory compliance" |

---

## 6. Domain Context: Hiring Bias

### Why Hiring?
- High-impact domain where AI bias has real consequences
- Well-documented historical biases (Amazon resume screening, 2018)
- Clear protected attributes defined by law (gender, race, age)
- Regulatory framework exists (EEOC, EU AI Act)

### Bias Sources in Hiring Data
1. **Historical bias**: Past hiring decisions reflect human prejudice
2. **Representation bias**: Underrepresented groups in training data
3. **Measurement bias**: Proxies for protected attributes (zip code → race)
4. **Label bias**: "Successful hire" definition may be biased

AEGIS AI addresses all four by auditing the trained model's outputs against multiple fairness criteria and providing actionable recommendations.
