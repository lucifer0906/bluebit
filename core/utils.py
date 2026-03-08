"""
AEGIS AI - Utility functions
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any


def load_dataset(path: str) -> pd.DataFrame:
    """Load a CSV dataset."""
    return pd.read_csv(path)


def get_sensitive_columns(df: pd.DataFrame) -> List[str]:
    """Identify potential sensitive attribute columns."""
    sensitive_keywords = {
        'gender', 'sex', 'race', 'ethnicity', 'age', 'religion',
        'disability', 'marital_status', 'nationality', 'sexual_orientation'
    }
    
    return [col for col in df.columns if col.lower() in sensitive_keywords]


def format_percentage(value: float) -> str:
    """Format a float as a percentage string."""
    return f"{value * 100:.1f}%"


def compute_group_statistics(df: pd.DataFrame, group_col: str, 
                            target_col: str) -> Dict[str, Any]:
    """Compute statistics per group for a target variable."""
    stats = {}
    for group in df[group_col].unique():
        mask = df[group_col] == group
        group_data = df[mask][target_col]
        stats[group] = {
            'count': int(mask.sum()),
            'mean': round(group_data.mean(), 4),
            'std': round(group_data.std(), 4),
            'positive_rate': round((group_data == 1).mean(), 4) if group_data.dtype in ['int64', 'float64'] else None,
        }
    return stats
