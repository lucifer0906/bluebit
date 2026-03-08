"""
AEGIS AI - Backend Configuration
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'tests', 'test_results')

# Default dataset
DEFAULT_DATASET = os.path.join(DATA_DIR, 'synthetic_hiring_data.csv')

# Fairness thresholds
DEMOGRAPHIC_PARITY_THRESHOLD = 0.8
EQUAL_OPPORTUNITY_THRESHOLD = 0.1
CALIBRATION_THRESHOLD = 0.25
TRANSPARENCY_THRESHOLD = 60
