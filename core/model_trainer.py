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
        
        Args:
            include_sensitive: If True, include gender/race as features (biased model).
                             If False, exclude them (potentially fairer model).
        """
        df = pd.read_csv(self.data_path)
        
        # Store sensitive features separately
        self.sensitive_features = df[['gender', 'race']].copy()
        
        # Encode categorical variables
        categorical_cols = ['gender', 'race']
        for col in categorical_cols:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        # Define features
        base_features = [
            'education_years', 'experience_years', 'skill_score',
            'interview_score', 'certification_count', 'project_count',
            'gpa', 'age'
        ]
        
        if include_sensitive:
            feature_cols = base_features + ['gender_encoded', 'race_encoded']
            self.feature_names = base_features + ['gender', 'race']
        else:
            feature_cols = base_features
            self.feature_names = base_features
        
        X = df[feature_cols].values
        y = df['hired'].values
        
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
    
    def train_all(self):
        """Train all models and return them."""
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()
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
