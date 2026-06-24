"""
Model training, evaluation, and selection module.
Implements Algorithm 3.1 (training pipeline) and Algorithm 3.3 (k-fold CV)
from Chapter 3.
Models: Linear Regression, Decision Tree, Random Forest, XGBoost.
"""

import os
import sys
import json
import numpy as np
import joblib
from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    MODEL_PIPELINE_PATH, METRICS_PATH, MODEL_DIR,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, BINARY_FEATURES,
    RF_PARAMS, XGB_PARAMS, CV_FOLDS, RANDOM_STATE
)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             CATEGORICAL_FEATURES),
            ('bin', 'passthrough', BINARY_FEATURES),
        ],
        remainder='drop'
    )


def build_models() -> dict:
    return {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(
            max_depth=10, min_samples_split=10, random_state=RANDOM_STATE
        ),
        'Random Forest': RandomForestRegressor(**RF_PARAMS),
        'XGBoost': xgb.XGBRegressor(**XGB_PARAMS),
    }


def evaluate_model(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
    return {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4),
        'MAPE': round(mape, 2),
    }


def cross_validate_model(pipeline: Pipeline, X, y, k: int = CV_FOLDS) -> dict:
    """Algorithm 3.3: K-Fold Cross-Validation."""
    kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        'r2': 'r2',
        'neg_mae': 'neg_mean_absolute_error',
        'neg_rmse': 'neg_root_mean_squared_error',
    }
    cv_results = cross_validate(pipeline, X, y, cv=kf, scoring=scoring,
                                return_train_score=False, n_jobs=-1)
    return {
        'cv_r2_mean': round(float(np.mean(cv_results['test_r2'])), 4),
        'cv_r2_std': round(float(np.std(cv_results['test_r2'])), 4),
        'cv_mae_mean': round(float(-np.mean(cv_results['test_neg_mae'])), 2),
        'cv_rmse_mean': round(float(-np.mean(cv_results['test_neg_rmse'])), 2),
    }


def train_and_evaluate(X_train, X_test, y_train, y_test) -> tuple:
    """Train all models and return metrics + best pipeline."""
    preprocessor = build_preprocessor()
    models = build_models()
    all_metrics = {}
    trained_pipelines = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model),
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = evaluate_model(y_test.values, y_pred)

        # Cross-validation on training data
        cv_stats = cross_validate_model(pipeline, X_train, y_train)
        metrics.update(cv_stats)

        all_metrics[name] = metrics
        trained_pipelines[name] = pipeline

        print(f"  MAE:  ₦{metrics['MAE']:>15,.0f}")
        print(f"  RMSE: ₦{metrics['RMSE']:>15,.0f}")
        print(f"  R²:   {metrics['R2']:.4f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        print(f"  CV R² (mean ± std): {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")

    # Select best model by CV R² (statistically proper: avoids test-set overfit)
    best_name = max(all_metrics, key=lambda k: all_metrics[k]['cv_r2_mean'])
    best_pipeline = trained_pipelines[best_name]
    print(f"\nBest model: {best_name} (R² = {all_metrics[best_name]['R2']:.4f})")

    return best_pipeline, best_name, all_metrics, trained_pipelines


def get_feature_names(pipeline: Pipeline) -> list:
    """Extract feature names after preprocessing."""
    preprocessor = pipeline.named_steps['preprocessor']
    num_names = NUMERICAL_FEATURES
    cat_names = list(preprocessor.named_transformers_['cat']
                     .get_feature_names_out(CATEGORICAL_FEATURES))
    bin_names = BINARY_FEATURES
    return num_names + cat_names + bin_names


def get_feature_importances(pipeline: Pipeline, top_n: int = 10) -> list:
    """Return top-N feature importances as list of (name, importance) tuples."""
    model = pipeline.named_steps['model']
    feature_names = get_feature_names(pipeline)

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        return []

    pairs = sorted(zip(feature_names, importances),
                   key=lambda x: x[1], reverse=True)
    top = pairs[:top_n]
    total = sum(imp for _, imp in top) or 1
    return [{'feature': name, 'importance': round(float(imp / total), 4)}
            for name, imp in top]


def compute_confidence_interval(pipeline: Pipeline, X_input, alpha: float = 0.10) -> dict:
    """
    Compute 90% confidence interval.
    For RF: use standard deviation of individual tree predictions.
    For other models: use ±15% of predicted value.
    """
    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocessor']
    X_transformed = preprocessor.transform(X_input)

    y_pred = model.predict(X_transformed)[0]

    if isinstance(model, RandomForestRegressor):
        tree_preds = np.array([tree.predict(X_transformed)[0]
                               for tree in model.estimators_])
        half_width = 1.645 * np.std(tree_preds)
    else:
        half_width = 0.15 * abs(y_pred)

    return {
        'lower': max(0, round(y_pred - half_width, -4)),
        'upper': round(y_pred + half_width, -4),
    }


def save_pipeline(pipeline: Pipeline, best_name: str, all_metrics: dict,
                  dataset_size: int, feature_count: int,
                  trained_pipelines: dict = None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PIPELINE_PATH)
    print(f"Pipeline saved to {MODEL_PIPELINE_PATH}")

    metadata = {
        'best_model': best_name,
        'model_comparison': all_metrics,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'dataset_size': dataset_size,
        'feature_count': feature_count,
        'model_version': f'{best_name.replace(" ", "_")}_v1',
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")
