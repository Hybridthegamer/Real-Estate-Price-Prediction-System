"""
Training entry point.
Executes the full offline model training pipeline (Algorithm 3.1, Chapter 3):
  generate data → preprocess → feature engineer → train → evaluate → save
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))

from config import RAW_DATA_PATH, METRICS_PATH
from src.data_preprocessing import (
    load_dataset, preprocess_dataset, split_data, save_processed_data
)
from src.feature_engineering import engineer_features
from src.model_training import train_and_evaluate, save_pipeline


def main():
    print("=" * 60)
    print("  Real Estate Price Prediction – Model Training Pipeline")
    print("=" * 60)

    # Step 1: Load dataset (generate if missing)
    if not os.path.exists(RAW_DATA_PATH):
        print("\nDataset not found. Generating synthetic data...")
        from generate_data import generate_dataset
        import pandas as pd
        from config import N_SAMPLES
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        df = generate_dataset(3000)
        df.to_csv(RAW_DATA_PATH, index=False)
        print(f"Dataset generated: {len(df)} records.")

    df = load_dataset()

    # Step 2: Preprocess
    print("\nPreprocessing dataset...")
    df = preprocess_dataset(df)
    save_processed_data(df)

    # Step 3: Feature engineering
    print("\nEngineering features...")
    df = engineer_features(df)
    print(f"Feature columns: {list(df.columns)}")

    # Step 4: Train/test split
    X_train, X_test, y_train, y_test = split_data(df)

    # Step 5: Train and evaluate all models
    print("\nTraining models...")
    best_pipeline, best_name, all_metrics, trained_pipelines = train_and_evaluate(
        X_train, X_test, y_train, y_test
    )

    # Step 6: Save best pipeline and metrics
    feature_count = X_train.shape[1]
    dataset_size = len(df)
    save_pipeline(best_pipeline, best_name, all_metrics,
                  dataset_size, feature_count, trained_pipelines)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best model : {best_name}")
    print(f"Test R²    : {all_metrics[best_name]['R2']:.4f}")
    print(f"Test MAE   : ₦{all_metrics[best_name]['MAE']:,.0f}")
    print(f"Test RMSE  : ₦{all_metrics[best_name]['RMSE']:,.0f}")
    print(f"Test MAPE  : {all_metrics[best_name]['MAPE']:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
