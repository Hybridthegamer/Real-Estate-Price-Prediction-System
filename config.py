import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'housing_data.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'processed_data.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PIPELINE_PATH = os.path.join(MODEL_DIR, 'model_pipeline.pkl')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
PREDICTION_LOG_PATH = os.path.join(LOG_DIR, 'prediction_log.json')

# Feature configuration
CATEGORICAL_FEATURES = ['neighbourhood', 'property_type']
NUMERICAL_FEATURES = [
    'sqft_living', 'property_age', 'bedrooms', 'bathrooms',
    'parking_spaces', 'lat', 'long', 'log_sqft', 'bedroom_bathroom_ratio'
]
BINARY_FEATURES = ['has_pool', 'is_gated', 'has_gym', 'renovation_flag']
TARGET_COLUMN = 'price'

# Model hyperparameters
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'n_jobs': -1
}

XGB_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}

# Training configuration
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 10

# Flask configuration
SECRET_KEY = os.environ.get('SECRET_KEY', 'realestate_fyp_2025_secret_key')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'admin123')
DEBUG = os.environ.get('FLASK_DEBUG', 'True') == 'True'
HOST = '0.0.0.0'
PORT = int(os.environ.get('PORT', 5000))

# Nigerian neighbourhood coordinates and price tiers
NEIGHBOURHOOD_COORDS = {
    # Lagos – Ultra Premium
    'Victoria Island': {'lat': 6.4281, 'long': 3.4219, 'tier': 'ultra_premium', 'base_per_sqm': 1_800_000},
    'Ikoyi': {'lat': 6.4444, 'long': 3.4422, 'tier': 'ultra_premium', 'base_per_sqm': 1_600_000},
    'Banana Island': {'lat': 6.4527, 'long': 3.4361, 'tier': 'ultra_premium', 'base_per_sqm': 2_500_000},
    # Lagos – Premium
    'Lekki Phase 1': {'lat': 6.4401, 'long': 3.5042, 'tier': 'premium', 'base_per_sqm': 750_000},
    'Lekki Phase 2': {'lat': 6.4573, 'long': 3.5387, 'tier': 'premium', 'base_per_sqm': 600_000},
    # Lagos – Mid-High
    'Ajah': {'lat': 6.4682, 'long': 3.5796, 'tier': 'mid_high', 'base_per_sqm': 300_000},
    'Gbagada': {'lat': 6.5483, 'long': 3.3876, 'tier': 'mid_high', 'base_per_sqm': 250_000},
    'Magodo': {'lat': 6.5917, 'long': 3.3880, 'tier': 'mid_high', 'base_per_sqm': 280_000},
    # Lagos – Standard
    'Surulere': {'lat': 6.5016, 'long': 3.3560, 'tier': 'standard', 'base_per_sqm': 150_000},
    'Yaba': {'lat': 6.5046, 'long': 3.3770, 'tier': 'standard', 'base_per_sqm': 180_000},
    'Ikeja': {'lat': 6.5956, 'long': 3.3382, 'tier': 'standard', 'base_per_sqm': 160_000},
    'Ogba': {'lat': 6.5836, 'long': 3.3447, 'tier': 'standard', 'base_per_sqm': 130_000},
    # Abuja – Ultra Premium
    'Maitama': {'lat': 9.0723, 'long': 7.4892, 'tier': 'ultra_premium', 'base_per_sqm': 1_200_000},
    'Asokoro': {'lat': 9.0467, 'long': 7.5190, 'tier': 'ultra_premium', 'base_per_sqm': 1_000_000},
    # Abuja – Premium
    'Wuse 2': {'lat': 9.0610, 'long': 7.4712, 'tier': 'premium', 'base_per_sqm': 600_000},
    'Jabi': {'lat': 9.0814, 'long': 7.4527, 'tier': 'premium', 'base_per_sqm': 500_000},
    # Abuja – Mid-High
    'Garki': {'lat': 9.0465, 'long': 7.4836, 'tier': 'mid_high', 'base_per_sqm': 350_000},
    'Gwarinpa': {'lat': 9.1186, 'long': 7.4144, 'tier': 'mid_high', 'base_per_sqm': 200_000},
    # Abuja – Standard
    'Kubwa': {'lat': 9.1395, 'long': 7.3491, 'tier': 'standard', 'base_per_sqm': 120_000},
    'Lugbe': {'lat': 9.0199, 'long': 7.4165, 'tier': 'standard', 'base_per_sqm': 100_000},
    # Port Harcourt – Premium
    'GRA Phase 1': {'lat': 4.8117, 'long': 7.0199, 'tier': 'premium', 'base_per_sqm': 500_000},
    'GRA Phase 2': {'lat': 4.8009, 'long': 7.0176, 'tier': 'premium', 'base_per_sqm': 400_000},
    # Port Harcourt – Mid
    'Trans Amadi': {'lat': 4.8494, 'long': 7.0356, 'tier': 'mid_high', 'base_per_sqm': 200_000},
    'Rumuola': {'lat': 4.8264, 'long': 7.0218, 'tier': 'mid_high', 'base_per_sqm': 180_000},
    # Port Harcourt – Standard
    'Diobu': {'lat': 4.8228, 'long': 6.9986, 'tier': 'standard', 'base_per_sqm': 100_000},
    'Eleme': {'lat': 4.7762, 'long': 7.0873, 'tier': 'standard', 'base_per_sqm': 80_000},
}

PROPERTY_TYPE_MULTIPLIERS = {
    'Detached House': 1.20,
    'Semi-Detached House': 1.00,
    'Apartment': 0.85,
    'Terrace': 0.90,
    'Bungalow': 0.95,
}

PROPERTY_TYPES = list(PROPERTY_TYPE_MULTIPLIERS.keys())
NEIGHBOURHOODS = list(NEIGHBOURHOOD_COORDS.keys())
