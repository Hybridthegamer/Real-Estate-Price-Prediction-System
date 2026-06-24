"""
Data loading, cleaning, and preprocessing pipeline.
Implements Algorithm 3.1 steps 1-4 from Chapter 3.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    RAW_DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE,
    PROCESSED_DATA_PATH
)


def load_dataset(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python generate_data.py"
        )
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        print(f"Removed {removed} duplicate rows")
    return df.reset_index(drop=True)


def impute_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values found:\n{missing[missing > 0]}")

    for col in numerical_cols:
        if df[col].isnull().any():
            fill_val = df[col].median() if strategy == 'median' else df[col].mean()
            df[col] = df[col].fillna(fill_val)

    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def remove_outliers(df: pd.DataFrame, method: str = 'IQR',
                    threshold: float = 1.5) -> pd.DataFrame:
    before = len(df)
    numeric_cols = [TARGET_COLUMN, 'sqft_living', 'sqft_lot']

    if method == 'IQR':
        mask = pd.Series([True] * len(df), index=df.index)
        for col in numeric_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                mask = mask & df[col].between(lower, upper)
        df = df[mask]

    removed = before - len(df)
    if removed:
        print(f"Removed {removed} outlier rows using {method} method")
    return df.reset_index(drop=True)


def validate_data_types(df: pd.DataFrame) -> pd.DataFrame:
    int_cols = ['bedrooms', 'bathrooms', 'floors', 'parking_spaces',
                'yr_built', 'yr_renovated', 'has_pool', 'is_gated',
                'has_gym', 'grade', 'condition']
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    float_cols = ['price', 'sqft_living', 'sqft_lot', 'lat', 'long']
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_duplicates(df)
    df = impute_missing_values(df)
    df = remove_outliers(df)
    df = validate_data_types(df)
    return df


def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def save_processed_data(df: pd.DataFrame, path: str = PROCESSED_DATA_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Processed data saved to {path}")
