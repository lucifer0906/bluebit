"""
AEGIS AI - Model Trainer
Trains multiple ML models on the hiring dataset for bias auditing.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
import joblib
import os


class ModelTrainer:
    """Train and manage ML models for hiring prediction."""
    
    MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    
    def __init__(self, data_path: str = None):
        if data_path is None:
            # Check for local recruitment dataset first
            local_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'dataset', 'data.csv'
            )
            if os.path.exists(local_path):
                data_path = local_path
            else:
                data_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    'data', 'synthetic_hiring_data.csv'
                )
        self.data_path = data_path
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.models = {}
        
    def load_and_prepare_data(self, include_sensitive: bool = True):
        """
        Load dataset and prepare features.
        """
        df = pd.read_csv(self.data_path)
        
        # Determine columns based on dataset
        if 'HiringDecision' in df.columns:
            # Recruitment Dataset (Local)
            target_col = 'HiringDecision'
            sensitive_cols = ['Gender', 'Age']
            categorical_cols = ['Gender']
            base_features = [c for c in df.columns if c not in [target_col] + sensitive_cols]
            self.sensitive_features = df[sensitive_cols].copy()
            # Ensure Gender is numeric
            if df['Gender'].dtype == object:
                df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        else:
            # Synthetic Dataset
            target_col = 'hired'
            sensitive_cols = ['gender', 'race']
            categorical_cols = ['gender', 'race']
            base_features = [
                'education_years', 'experience_years', 'skill_score',
                'interview_score', 'certification_count', 'project_count',
                'gpa', 'age'
            ]
            self.sensitive_features = df[sensitive_cols].copy()

        # Encode categorical variables
        for col in categorical_cols:
            if df[col].dtype == object or col not in df.columns:
                le = LabelEncoder()
                # Handle potential missing cols in synthetic
                if col in df.columns:
                    df[f'{col}_encoded'] = le.fit_transform(df[col])
                    self.label_encoders[col] = le
        
        if include_sensitive:
            feature_cols = base_features + sensitive_cols
            self.feature_names = base_features + sensitive_cols
        else:
            feature_cols = base_features
            self.feature_names = base_features
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test, \
        self.sensitive_train, self.sensitive_test = train_test_split(
            X, y, self.sensitive_features, test_size=0.3, random_state=42
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_logistic_regression(self):
        """Train a Logistic Regression model."""
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(self.X_train, self.y_train)
        self.models['logistic_regression'] = model
        return model
    
    def train_random_forest(self):
        """Train a Random Forest model."""
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
        model.fit(self.X_train, self.y_train)
        self.models['random_forest'] = model
        return model
    
    def train_xgboost(self):
        """Train an XGBoost model."""
        model = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric='logloss'
        )
        model.fit(self.X_train, self.y_train)
        self.models['xgboost'] = model
        return model

    def train_adversarial_debiaser(self):
        """Train an Adversarial Debiasing model."""
        from core.debiasing import AdversarialDebiaser
        
        # Determine sensitive attribute name for the debiaser
        # If multiple, use the first one (usually Gender)
        sensitive_attr = self.sensitive_features.columns[0]
        
        model = AdversarialDebiaser(
            feature_names=self.feature_names,
            privileged_groups=[{sensitive_attr: 1}],
            unprivileged_groups=[{sensitive_attr: 0}]
        )
        
        model.fit(self.X_train, self.y_train)
        self.models['adversarial_debiaser'] = model
        return model
    
    def train_all(self, include_debiased: bool = True):
        """Train all models and return them."""
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()
        if include_debiased:
            self.train_adversarial_debiaser()
        return self.models
    
    def save_models(self):
        """Save all trained models to disk."""
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        
        for name, model in self.models.items():
            path = os.path.join(self.MODELS_DIR, f'{name}.joblib')
            joblib.dump(model, path)
            print(f"  Saved {name} → {path}")
        
        # Save scaler and encoders
        joblib.dump(self.scaler, os.path.join(self.MODELS_DIR, 'scaler.joblib'))
        joblib.dump(self.label_encoders, os.path.join(self.MODELS_DIR, 'label_encoders.joblib'))
        joblib.dump(self.feature_names, os.path.join(self.MODELS_DIR, 'feature_names.joblib'))
        
    def load_model(self, model_name: str):
        """Load a saved model from disk."""
        path = os.path.join(self.MODELS_DIR, f'{model_name}.joblib')
        return joblib.load(path)
    
    def get_predictions(self, model_name: str = None, model=None):
        """Get predictions and probabilities for the test set."""
        if model is None:
            model = self.models.get(model_name)
        
        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)[:, 1]
        
        return y_pred, y_prob


def train_all_models():
    """Convenience function to train and save all models."""
    print("🔧 Training hiring prediction models...")
    trainer = ModelTrainer()
    trainer.load_and_prepare_data(include_sensitive=True)
    trainer.train_all()
    trainer.save_models()
    print("✅ All models trained and saved!")
    return trainer


if __name__ == "__main__":
    train_all_models()
