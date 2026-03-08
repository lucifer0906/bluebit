import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from core.bias_detector import BiasDetector

def run_audit():
    print("Loading FairJob dataset from Hugging Face...")
    # Load a representative subset for speed
    df = pd.read_csv("hf://datasets/criteo/FairJob/fairjob.csv.gz", nrows=5000)
    
    # Define features
    target = 'click'
    sensitive_col = 'protected_attribute'
    
    # Drop columns that are not useful for prediction or might be identifiers
    cols_to_drop = [target, sensitive_col, 'user_id', 'impression_id', 'product_id']
    X = df.drop(columns=cols_to_drop)
    y = df[target]
    
    sensitive_features = df[[sensitive_col]]
    feature_names = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X.values, y.values, sensitive_features, test_size=0.2, random_state=42
    )
    
    print(f"Training Baseline Model (Random Forest)...")
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    print("Initializing BiasDetector and running audit...")
    detector = BiasDetector()
    
    # Run the audit
    audit_report = detector.audit_model(
        model=model,
        model_name="FairJob_Baseline_RF",
        X_train=X_train,
        X_test=X_test,
        y_true=y_test,
        sensitive_features=sens_test,
        feature_names=feature_names,
        sensitive_attributes=[sensitive_col]
    )
    
    # Display results
    print("\n--- AUDIT SUMMARY ---")
    verdict = audit_report['overall_verdict']
    print(f"Model: {audit_report['model_name']}")
    print(f"Fairness Score: {verdict['score']}/100")
    print(f"Grade: {verdict['grade']} ({verdict['label']})")
    print(f"Passed: {verdict['passed']}")
    
    print("\n--- KEY FINDINGS ---")
    for rec in audit_report['recommendations']:
        print(f"[{rec['severity']}] {rec['category']}: {rec['finding']}")
        print(f"Recommendation: {rec['recommendation']}\n")

if __name__ == "__main__":
    if not os.path.exists('scripts'):
        os.makedirs('scripts')
    run_audit()
