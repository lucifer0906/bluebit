"""
AEGIS AI - Debiasing Module
Implements Adversarial Debiasing to mitigate bias in ML models.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.inprocessing import AdversarialDebiasing
from sklearn.base import BaseEstimator, ClassifierMixin

class AdversarialDebiaser(BaseEstimator, ClassifierMixin):
    """
    Wrapper for AIF360 AdversarialDebiasing to make it scikit-learn compatible.
    """
    
    def __init__(self, privileged_groups, unprivileged_groups, 
                 sensitive_attribute, target_attribute,
                 adversary_loss_weight=0.1, num_epochs=50, batch_size=64,
                 feature_names=None):
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups
        self.sensitive_attribute = sensitive_attribute
        self.target_attribute = target_attribute
        self.adversary_loss_weight = adversary_loss_weight
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.feature_names = feature_names
        self.sess = None
        self.debiaser = None
        
    def fit(self, X, y):
        # Prepare AIF360 Dataset
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            if self.feature_names is None:
                self.feature_names = X.columns.tolist()
            # Ensure indices match
            df = df.reset_index(drop=True)
        else:
            cols = self.feature_names if self.feature_names else [f'f{i}' for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=cols)
            
        if isinstance(y, (pd.Series, pd.DataFrame)):
            df[self.target_attribute] = y.values
        else:
            df[self.target_attribute] = y
            
        # Sensitive attribute must be in df
        if self.sensitive_attribute not in df.columns:
            # If not found, assume it's the first column (legacy/fallback)
            df.rename(columns={df.columns[0]: self.sensitive_attribute}, inplace=True)
        
        # Convert to AIF360 format
        dataset = BinaryLabelDataset(
            df=df,
            label_names=[self.target_attribute],
            protected_attribute_names=[self.sensitive_attribute],
            favorable_label=1,
            unfavorable_label=0
        )
        
        # Initialize and train
        tf.compat.v1.reset_default_graph()
        self.sess = tf.compat.v1.Session()
        
        self.debiaser = AdversarialDebiasing(
            privileged_groups=self.privileged_groups,
            unprivileged_groups=self.unprivileged_groups,
            scope_name='debiased_model',
            debias=True,
            adversary_loss_weight=self.adversary_loss_weight,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            sess=self.sess
        )
        
        self.debiaser.fit(dataset)
        return self
        
    def predict(self, X):
        # Convert X to AIF360 format
        if isinstance(X, pd.DataFrame):
            df = X.copy().reset_index(drop=True)
        else:
            cols = self.feature_names if self.feature_names else [f'f{i}' for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=cols)
            
        # AIF360 needs a dummy label
        df[self.target_attribute] = 0
        
        # Ensure sensitive attribute name is correct if we relied on indices
        if self.sensitive_attribute not in df.columns and self.feature_names is None:
            df.rename(columns={df.columns[0]: self.sensitive_attribute}, inplace=True)

        dataset = BinaryLabelDataset(
            df=df,
            label_names=[self.target_attribute],
            protected_attribute_names=[self.sensitive_attribute],
            favorable_label=1,
            unfavorable_label=0
        )
        
        pred_dataset = self.debiaser.predict(dataset)
        return pred_dataset.labels.flatten()
        
    def predict_proba(self, X):
        # Similar to predict, but return scores
        if isinstance(X, pd.DataFrame):
            df = X.copy().reset_index(drop=True)
        else:
            cols = self.feature_names if self.feature_names else [f'f{i}' for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=cols)
            
        df[self.target_attribute] = 0
        
        if self.sensitive_attribute not in df.columns and self.feature_names is None:
            df.rename(columns={df.columns[0]: self.sensitive_attribute}, inplace=True)

        dataset = BinaryLabelDataset(
            df=df,
            label_names=[self.target_attribute],
            protected_attribute_names=[self.sensitive_attribute],
            favorable_label=1,
            unfavorable_label=0
        )
        
        pred_dataset = self.debiaser.predict(dataset)
        # AdversarialDebialing scores are in .scores
        probs = pred_dataset.scores.flatten()
        return np.vstack([1-probs, probs]).T
        
    def close(self):
        if self.sess:
            self.sess.close()
