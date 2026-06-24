"""
Runtime prediction inference pipeline.
Implements Algorithm 3.2 from Chapter 3.
"""

import os
import sys
import json
import uuid
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    MODEL_PIPELINE_PATH, PREDICTION_LOG_PATH,
    NEIGHBOURHOOD_COORDS, NEIGHBOURHOODS, PROPERTY_TYPES,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES
)
from src.feature_engineering import engineer_inference_features
from src.model_training import get_feature_importances, compute_confidence_interval

_pipeline = None


def load_pipeline():
    global _pipeline
    if _pipeline is None:
        if not os.path.exists(MODEL_PIPELINE_PATH):
            raise FileNotFoundError(
                "Model pipeline not found. Run: python train.py"
            )
        _pipeline = joblib.load(MODEL_PIPELINE_PATH)
    return _pipeline


def validate_input(data: dict) -> list:
    """Return list of validation error strings (empty = valid)."""
    errors = []

    required_fields = [
        'neighbourhood', 'property_type', 'sqft_living',
        'bedrooms', 'bathrooms', 'parking_spaces', 'yr_built',
        'has_pool', 'is_gated', 'has_gym',
    ]
    for field in required_fields:
        if field not in data or data[field] is None or str(data[field]).strip() == '':
            errors.append(f"'{field}' is required.")

    if errors:
        return errors

    if data.get('neighbourhood') not in NEIGHBOURHOODS:
        errors.append(f"Invalid neighbourhood: '{data.get('neighbourhood')}'.")

    if data.get('property_type') not in PROPERTY_TYPES:
        errors.append(f"Invalid property type: '{data.get('property_type')}'.")

    try:
        sqft = float(data['sqft_living'])
        if not (10 <= sqft <= 2000):
            errors.append("Floor area must be between 10 and 2,000 m².")
    except (ValueError, TypeError):
        errors.append("Floor area must be a number.")

    try:
        beds = int(data['bedrooms'])
        if not (1 <= beds <= 20):
            errors.append("Bedrooms must be between 1 and 20.")
    except (ValueError, TypeError):
        errors.append("Bedrooms must be an integer.")

    try:
        baths = int(data['bathrooms'])
        if not (1 <= baths <= 15):
            errors.append("Bathrooms must be between 1 and 15.")
    except (ValueError, TypeError):
        errors.append("Bathrooms must be an integer.")

    try:
        park = int(data['parking_spaces'])
        if not (0 <= park <= 10):
            errors.append("Parking spaces must be between 0 and 10.")
    except (ValueError, TypeError):
        errors.append("Parking spaces must be an integer.")

    try:
        yr = int(data['yr_built'])
        if not (1900 <= yr <= 2025):
            errors.append("Year of construction must be between 1900 and 2025.")
    except (ValueError, TypeError):
        errors.append("Year of construction must be a valid year.")

    return errors


def _build_feature_row(data: dict) -> pd.DataFrame:
    """Map validated user input to model feature row."""
    nb = data['neighbourhood']
    coords = NEIGHBOURHOOD_COORDS[nb]

    raw = {
        'neighbourhood': nb,
        'property_type': data['property_type'],
        'sqft_living': float(data['sqft_living']),
        'sqft_lot': float(data.get('sqft_lot', float(data['sqft_living']) * 2.0)),
        'bedrooms': int(data['bedrooms']),
        'bathrooms': int(data['bathrooms']),
        'floors': int(data.get('floors', 1)),
        'parking_spaces': int(data['parking_spaces']),
        'yr_built': int(data['yr_built']),
        'yr_renovated': int(data.get('yr_renovated', 0)),
        'has_pool': int(data.get('has_pool', 0)),
        'is_gated': int(data.get('is_gated', 0)),
        'has_gym': int(data.get('has_gym', 0)),
        'grade': int(data.get('grade', 7)),
        'condition': int(data.get('condition', 3)),
        'lat': coords['lat'],
        'long': coords['long'],
    }

    engineered = engineer_inference_features(raw)

    # Ensure all expected columns are present
    all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
    for feat in all_features:
        if feat not in engineered:
            engineered[feat] = 0

    return pd.DataFrame([engineered])


def predict(data: dict) -> dict:
    """
    Algorithm 3.2: Full inference pipeline.
    Returns predicted price, confidence interval, and feature importances.
    """
    errors = validate_input(data)
    if errors:
        return {'status': 'error', 'errors': errors}

    pipeline = load_pipeline()
    X = _build_feature_row(data)

    predicted_price = float(pipeline.predict(X)[0])
    predicted_price = max(5_000_000, predicted_price)

    ci = compute_confidence_interval(pipeline, X)
    importances = get_feature_importances(pipeline, top_n=10)

    # Map engineered feature names to human-readable labels
    label_map = {
        'sqft_living': 'Floor Area (m²)',
        'property_age': 'Property Age (yrs)',
        'bedrooms': 'No. of Bedrooms',
        'bathrooms': 'No. of Bathrooms',
        'parking_spaces': 'Parking Spaces',
        'lat': 'Latitude (Location)',
        'long': 'Longitude (Location)',
        'log_sqft': 'Log Floor Area',
        'bedroom_bathroom_ratio': 'Bed/Bath Ratio',
        'renovation_flag': 'Recently Renovated',
        'has_pool': 'Swimming Pool',
        'is_gated': 'Gated Estate',
        'has_gym': 'Gym Facility',
    }
    for item in importances:
        raw_name = item['feature']
        for prefix in ['neighbourhood_', 'property_type_']:
            if raw_name.startswith(prefix):
                item['feature'] = raw_name.replace('_', ' ').title()
                break
        else:
            item['feature'] = label_map.get(raw_name, raw_name.replace('_', ' ').title())

    response = {
        'status': 'success',
        'predicted_price': round(predicted_price, -4),
        'predicted_price_formatted': _format_naira(predicted_price),
        'confidence_interval': {
            'lower': ci['lower'],
            'upper': ci['upper'],
            'lower_formatted': _format_naira(ci['lower']),
            'upper_formatted': _format_naira(ci['upper']),
        },
        'feature_importances': importances,
    }

    _log_prediction(data, response)
    return response


def _format_naira(amount: float) -> str:
    return f"₦{amount:,.0f}"


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _log_prediction(input_data: dict, response: dict):
    """Append prediction event to prediction_log.json."""
    os.makedirs(os.path.dirname(PREDICTION_LOG_PATH), exist_ok=True)

    log_entry = {
        'id': str(uuid.uuid4()),
        'timestamp': datetime.now().isoformat(),
        'input': {k: v for k, v in input_data.items()
                  if k not in ('lat', 'long')},  # anonymise coords
        'predicted_price': float(response.get('predicted_price', 0)),
        'conf_lower': float(response.get('confidence_interval', {}).get('lower', 0)),
        'conf_upper': float(response.get('confidence_interval', {}).get('upper', 0)),
        'model_version': _get_model_version(),
    }

    existing = []
    if os.path.exists(PREDICTION_LOG_PATH):
        try:
            with open(PREDICTION_LOG_PATH) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = []

    existing.append(log_entry)
    # Keep last 500 entries
    existing = existing[-500:]

    with open(PREDICTION_LOG_PATH, 'w') as f:
        json.dump(existing, f, indent=2, cls=_NumpyEncoder)


def _get_model_version() -> str:
    from config import METRICS_PATH
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH) as f:
                meta = json.load(f)
            return meta.get('model_version', 'unknown')
        except Exception:
            pass
    return 'unknown'


def get_recent_predictions(n: int = 20) -> list:
    if not os.path.exists(PREDICTION_LOG_PATH):
        return []
    try:
        with open(PREDICTION_LOG_PATH) as f:
            logs = json.load(f)
        return logs[-n:][::-1]
    except (json.JSONDecodeError, IOError):
        return []
