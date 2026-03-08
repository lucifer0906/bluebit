import sys
import os
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from core.bias_detector import BiasDetector
from core.debiasing import AdversarialDebiaser
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def train_and_audit():
    print("Loading local recruitment dataset from dataset folder...")
    # Load the local dataset
    dataset_path = os.path.join("dataset", "data.csv")
    df = pd.read_csv(dataset_path)
    
    # Handle missing values
    df = df.dropna()
    
    # Column Mappings for the local dataset
    target = 'HiringDecision'
    sensitive_col = 'Gender'
    
    # Preprocessing
    # Ensure Gender is numeric if it's not (e.g. 'Male'/'Female')
    if df[sensitive_col].dtype == object:
        df[sensitive_col] = df[sensitive_col].map({'Male': 1, 'Female': 0})
    
    X_df = df.drop(columns=[target])
    y = df[target]
    
    feature_names = X_df.columns.tolist()
    
    # Split data
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    
    # Scale features (Critical for Adversarial Debiasing)
    scaler = StandardScaler()
    # We want to keep DataFrames if possible, or reconstructed them
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_df), 
        columns=X_train_df.columns, 
        index=X_train_df.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_df), 
        columns=X_test_df.columns, 
        index=X_test_df.index
    )
    
    # Define privileged/unprivileged groups for AIF360
    # In FairJob, protected_attribute=1 might be privileged, 0 unprivileged (or vice versa)
    # Let's assume 1 is privileged for now based on common patterns, or check parity
    privileged_groups = [{sensitive_col: 1}]
    unprivileged_groups = [{sensitive_col: 0}]
    
    print(f"Training Debiased Model (Adversarial Debiasing)...")
    # Wrap in our scikit-learn compatible class
    debiaser = AdversarialDebiaser(
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups,
        sensitive_attribute=sensitive_col,
        target_attribute=target,
        adversary_loss_weight=0.1,
        num_epochs=20,
        batch_size=128,
        feature_names=feature_names
    )
    
    # Train
    debiaser.fit(X_train_scaled, y_train)
    
    print("Initializing BiasDetector and running audit on Debiased Model...")
    detector = BiasDetector()
    
    # Prepare sensitive_features for the audit tool
    sens_test = X_test_scaled[[sensitive_col]]
    
    # Run the audit
    audit_report = detector.audit_model(
        model=debiaser,
        model_name="FairJob_Debiased_Adversarial",
        X_train=X_train_scaled.values,
        X_test=X_test_scaled.values,
        y_true=y_test.values,
        sensitive_features=sens_test,
        feature_names=feature_names,
        sensitive_attributes=[sensitive_col]
    )
    
    # Display results
    print("\n--- DEBIASED AUDIT SUMMARY ---")
    verdict = audit_report['overall_verdict']
    print(f"Model: {audit_report['model_name']}")
    print(f"Fairness Score: {verdict['score']}/100")
    print(f"Grade: {verdict['grade']} ({verdict['label']})")
    print(f"Passed: {verdict['passed']}")
    
    print("\n--- KEY FINDINGS ---")
    for rec in audit_report['recommendations']:
        print(f"[{rec['severity']}] {rec['category']}: {rec['finding']}")
        print(f"Recommendation: {rec['recommendation']}\n")
        
    debiaser.close()

if __name__ == "__main__":
    try:
        train_and_audit()
    except Exception as e:
        print(f"Error during training/audit: {e}")
        import traceback
        traceback.print_exc()
