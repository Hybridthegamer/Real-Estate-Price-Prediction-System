"""
Feature engineering module.
Implements Algorithm 3.1 ENGINEER_FEATURES step from Chapter 3.
Creates derived features: property_age, renovation_flag, log_sqft,
bedroom_bathroom_ratio from raw inputs.
"""

import numpy as np
import pandas as pd

CURRENT_YEAR = 2025


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Property age: more informative than raw year built
    df['property_age'] = CURRENT_YEAR - df['yr_built']
    df['property_age'] = df['property_age'].clip(lower=0)

    # Renovation flag: binary indicator derived from yr_renovated
    if 'yr_renovated' in df.columns:
        df['renovation_flag'] = (df['yr_renovated'] > 0).astype(int)
    else:
        df['renovation_flag'] = 0

    # Log-transformed floor area reduces skewness
    df['log_sqft'] = np.log1p(df['sqft_living'])

    # Bedroom-to-bathroom ratio captures space efficiency
    df['bedroom_bathroom_ratio'] = df['bedrooms'] / (df['bathrooms'].replace(0, 0.5))

    # Drop raw temporal columns no longer needed after engineering
    cols_to_drop = [c for c in ['yr_built', 'yr_renovated'] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    return df


def engineer_inference_features(data: dict) -> dict:
    """Apply feature engineering to a single inference input dict."""
    data = dict(data)

    yr_built = int(data.get('yr_built', CURRENT_YEAR - 10))
    yr_renovated = int(data.get('yr_renovated', 0))
    sqft_living = float(data.get('sqft_living', 100))
    bedrooms = float(data.get('bedrooms', 3))
    bathrooms = float(data.get('bathrooms', 2))

    data['property_age'] = max(0, CURRENT_YEAR - yr_built)
    data['renovation_flag'] = 1 if yr_renovated > 0 else 0
    data['log_sqft'] = np.log1p(sqft_living)
    data['bedroom_bathroom_ratio'] = bedrooms / max(bathrooms, 0.5)

    data.pop('yr_built', None)
    data.pop('yr_renovated', None)

    return data
