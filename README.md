# EstimateNG – Real Estate Price Prediction System

An AI-powered web application that delivers instant, data-driven residential
property price estimates for the Nigerian market using an ensemble of
supervised machine learning algorithms.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Requirements](#2-system-requirements)
3. [Installation & Setup](#3-installation--setup)
4. [Running the Application](#4-running-the-application)
5. [Directory Structure](#5-directory-structure)
6. [Tool Stack & Rationale](#6-tool-stack--rationale)
7. [Machine Learning Pipeline](#7-machine-learning-pipeline)
8. [API Reference](#8-api-reference)
9. [Admin Panel](#9-admin-panel)
10. [Model Performance](#10-model-performance)

---

## 1. Project Overview

EstimateNG addresses the subjectivity, cost, and inefficiency of manual real
estate valuation in Nigeria by providing an automated valuation model (AVM)
accessible via a web browser. Users enter property attributes — location,
floor area, bedrooms, bathrooms, amenities — and the system returns a
predicted market price in Nigerian Naira alongside:

- A **90% confidence interval** derived from ensemble tree-level variance
- A **feature importance bar chart** (Chart.js) explaining which attributes
  most influenced the estimate

The system was designed and implemented as a Final Year Project in Computer
Science. Chapters 1–3 of the accompanying thesis detail the background,
literature review, system requirements, database schema, UML diagrams, and
algorithmic pseudocode on which this implementation is based.

---

## 2. System Requirements

| Component | Minimum Version |
|-----------|----------------|
| Python | 3.9+ |
| pip | 23+ |
| RAM | 4 GB (8 GB recommended for training) |
| Disk | 500 MB free |
| OS | Windows 10 / macOS 11+ / Ubuntu 20.04+ |
| Browser | Chrome 100+, Firefox 100+, Edge 100+, Safari 15+ |

> **Note:** A GPU is not required. All training runs on CPU in under
> 5 minutes on a standard laptop.

---

## 3. Installation & Setup

### Step 1 – Clone the repository

```bash
git clone https://github.com/hybridthegamer/real-estate-price-prediction-system.git
cd real-estate-price-prediction-system
```

### Step 2 – Create and activate a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 – Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 4 – Generate the dataset

```bash
python generate_data.py
```

This creates `data/raw/housing_data.csv` — a synthetic dataset of 3,000
Nigerian residential property records with realistic price distributions
across Lagos, Abuja, and Port Harcourt.

### Step 5 – Train the machine learning models

```bash
python train.py
```

This runs the full training pipeline (Algorithm 3.1 from Chapter 3):

1. Loads and preprocesses the dataset (deduplication, imputation, IQR
   outlier removal)
2. Applies feature engineering (property age, log floor area,
   bedroom/bathroom ratio, renovation flag)
3. Trains four algorithms: Linear Regression, Decision Tree,
   Random Forest, and XGBoost
4. Evaluates each model using MAE, RMSE, R², MAPE, and 10-fold CV
5. Selects the best model by cross-validated R²
6. Serialises the winner + fitted preprocessor to
   `models/model_pipeline.pkl`
7. Saves comparison metrics to `models/metrics.json`

Expected training time: **2–5 minutes** on a modern CPU.

---

## 4. Running the Application

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

> On first launch, if `models/model_pipeline.pkl` is missing, the app
> automatically triggers the training pipeline before serving requests.

### Environment variables (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | `realestate_fyp_2025_secret_key` | Flask session key |
| `ADMIN_PASSWORD` | `admin123` | Admin panel password |
| `PORT` | `5000` | Server port |
| `FLASK_DEBUG` | `True` | Debug mode (set `False` in production) |

---

## 5. Directory Structure

```
Real-Estate-Price-Prediction-System/
├── app.py                    # Flask application entry point
├── train.py                  # CLI training script
├── generate_data.py          # Synthetic dataset generator
├── config.py                 # Centralised configuration & constants
├── requirements.txt
├── README.md
│
├── src/
│   ├── data_preprocessing.py # Load, clean, split dataset
│   ├── feature_engineering.py# Derived feature creation
│   ├── model_training.py     # Train, evaluate, save models
│   └── prediction_engine.py  # Runtime inference pipeline
│
├── data/
│   ├── raw/housing_data.csv  # Raw synthetic dataset
│   └── processed/            # Cleaned dataset (generated)
│
├── models/
│   ├── model_pipeline.pkl    # Serialised best model + preprocessor
│   └── metrics.json          # Model comparison metrics
│
├── templates/
│   ├── base.html             # Shared layout + navbar
│   ├── index.html            # Landing page
│   ├── predict.html          # Prediction form + results
│   ├── admin.html            # Admin panel
│   └── admin_login.html      # Admin login page
│
├── static/
│   ├── css/style.css         # Custom stylesheet
│   └── js/main.js            # Form handler + Chart.js rendering
│
└── logs/
    └── prediction_log.json   # Inference audit log (auto-created)
```

---

## 6. Tool Stack & Rationale

### Python 3.9+

Python is the dominant language in the data science and machine learning
ecosystem. Its rich library support (scikit-learn, pandas, numpy),
first-class Flask integration, and readable syntax make it the natural
choice for combining a data pipeline with a web server in a single
codebase. PEP 8 coding standards are followed throughout.

### Flask (Web Framework)

Flask is a lightweight WSGI micro-framework that is straightforward to
learn and deploy. Unlike full-stack frameworks such as Django, Flask does
not impose an ORM or fixed project layout — this aligns well with the
system's architecture, where data is handled by pandas/sklearn rather than
a relational database. Flask's native Jinja2 templating and RESTful routing
allow clean separation between the ML inference code and the HTTP layer.

### scikit-learn

scikit-learn provides a consistent, well-documented API for all four
regression algorithms benchmarked in this study (Linear Regression,
Decision Tree, Random Forest), as well as the `Pipeline` and
`ColumnTransformer` utilities that bundle preprocessing and model
inference into a single serialisable object. This ensures that the exact
transformations fitted on training data are reapplied identically at
runtime — a critical requirement for correct inference.

### XGBoost

XGBoost (Extreme Gradient Boosting) is selected as the primary model
based on its superior cross-validated R² (0.955) and lowest MAPE (14%)
across the four algorithms evaluated. XGBoost's regularised gradient
boosting reduces overfitting compared to vanilla gradient boosting,
handles numerical and encoded categorical features natively, and provides
feature importance estimates via `feature_importances_`. Its scikit-learn
API wrapper integrates seamlessly into the Pipeline architecture.

### Pandas & NumPy

Pandas provides the DataFrame abstraction for all data loading,
cleaning, transformation, and exploratory analysis steps. NumPy underpins
all numerical operations (IQR outlier detection, log transforms,
confidence interval calculations). Together they fulfil the data
manipulation objectives stated in Chapter 1, Objective 1.

### Matplotlib & Seaborn

Matplotlib and Seaborn are used during the offline training phase to
generate exploratory data visualisations (distribution plots, correlation
heatmaps, model comparison bar charts) that inform the data preprocessing
and feature engineering decisions documented in Chapter 4. These
libraries are not used at inference time.

### Joblib

Joblib serialises the fitted scikit-learn Pipeline (preprocessor +
model) to a single `.pkl` file. At application startup the pipeline is
loaded once into memory, ensuring sub-second inference latency for
subsequent prediction requests.

### Bootstrap 5

Bootstrap 5 provides the responsive CSS grid and component library that
makes the interface compatible with all modern browsers and screen sizes
(Chapter 3, §3.3.3 Compatibility requirement). The system's colour scheme
(primary `#1F3864` deep blue, accent `#F5A623` amber) is layered on top
via a custom `style.css` override.

### Chart.js

Chart.js renders the horizontal bar chart that visualises feature
importances in the result view. It was chosen for its small footprint
(single CDN script), zero-dependency Chart objects, and first-class
responsive canvas support — no server-side rendering or extra backend
calls are required.

### Google Fonts (Roboto / Roboto Slab)

Roboto is used for body text and Roboto Slab for headings, following the
typographic specification in Chapter 3, §3.12, providing a clean and
professional aesthetic consistent with the system's valuation context.

---

## 7. Machine Learning Pipeline

The pipeline implements the three algorithms specified in Chapter 3, §3.7:

### Algorithm 3.1 – Offline Training Pipeline

```
LOAD dataset → REMOVE_DUPLICATES → IMPUTE_MISSING (median)
→ REMOVE_OUTLIERS (IQR, threshold=1.5)
→ ENGINEER_FEATURES:
    property_age = 2025 - yr_built
    renovation_flag = (yr_renovated > 0)
    log_sqft = log(sqft_living + 1)
    bedroom_bathroom_ratio = bedrooms / bathrooms
→ TRAIN_TEST_SPLIT (80/20)
→ FOR model IN [LR, DT, RF, XGBoost]:
    FIT Pipeline(ColumnTransformer, model)
    EVALUATE(MAE, RMSE, R², MAPE)
    CROSS_VALIDATE(k=10)
→ SELECT_BEST(criterion=CV_R²)
→ SERIALIZE pipeline → model_pipeline.pkl
```

### Algorithm 3.2 – Runtime Inference Pipeline

```
RECEIVE user_input (JSON POST)
→ VALIDATE_INPUT (field presence, type, range)
→ LOOKUP lat/long from neighbourhood
→ ENGINEER_FEATURES (same as training)
→ pipeline.predict(X)  ← ColumnTransformer + XGBoost
→ COMPUTE_CONFIDENCE_INTERVAL (±15% of predicted value)
→ GET_FEATURE_IMPORTANCES (top 10)
→ LOG_PREDICTION (prediction_log.json)
→ RETURN JSON response
```

### Algorithm 3.3 – K-Fold Cross-Validation (k=10)

Used during model selection to obtain robust, unbiased estimates of
generalisation performance. Mean CV R² and its standard deviation are
reported alongside test-set metrics in `models/metrics.json`.

---

## 8. API Reference

### `POST /api/predict`

Returns a property price prediction.

**Request body (JSON):**

```json
{
  "neighbourhood":  "Lekki Phase 1",
  "property_type":  "Detached House",
  "sqft_living":    300,
  "bedrooms":       4,
  "bathrooms":      3,
  "parking_spaces": 2,
  "yr_built":       2018,
  "has_pool":       1,
  "is_gated":       1,
  "has_gym":        0
}
```

**Success response (200):**

```json
{
  "status": "success",
  "predicted_price": 348380032,
  "predicted_price_formatted": "₦348,380,032",
  "confidence_interval": {
    "lower": 296120000,
    "upper": 400640000,
    "lower_formatted": "₦296,120,000",
    "upper_formatted": "₦400,640,000"
  },
  "feature_importances": [
    { "feature": "Neighbourhood Lekki Phase 1", "importance": 0.2638 },
    { "feature": "Floor Area (m²)",             "importance": 0.1349 }
  ]
}
```

**Error response (400):**

```json
{
  "status": "error",
  "errors": ["Floor area must be between 10 and 2,000 m²."]
}
```

### `GET /api/metrics`

Returns the model comparison metrics stored in `models/metrics.json`.

---

## 9. Admin Panel

Access the admin panel at **http://localhost:5000/admin**.

Default password: `admin123`
(Change via the `ADMIN_PASSWORD` environment variable in production.)

**Admin capabilities:**

| Feature | Description |
|---------|-------------|
| Model Summary | Deployed model name, training date, dataset size |
| Model Comparison Table | R² and MAPE for all four algorithms |
| Recent Predictions | Last 20 inference requests (anonymised) |
| Upload Dataset | Replace training CSV with a new one |
| Retrain Model | Trigger full pipeline re-run on the server |

---

## 10. Model Performance

Results on a 20% holdout test set (533 records) from the synthetic
Nigerian housing dataset:

| Algorithm | Test R² | Test MAE | MAPE | CV R² (mean±std) |
|-----------|--------:|--------:|-----:|----------------:|
| Linear Regression | 0.9518 | ₦11.9M | 39.3% | 0.9487 ± 0.0076 |
| Decision Tree | 0.8392 | ₦15.2M | 27.1% | 0.8977 ± 0.0264 |
| Random Forest | 0.8948 | ₦11.3M | 17.2% | 0.9275 ± 0.0264 |
| **XGBoost** ✓ | **0.9369** | **₦9.2M** | **14.1%** | **0.9550 ± 0.0128** |

XGBoost is selected as the production model based on its highest
cross-validated R² (0.9550) and lowest MAPE (14.1%). The lower MAPE
relative to Linear Regression reflects XGBoost's superior handling of
the non-linear interactions between location tier, property type, and
floor area — relationships that OLS cannot capture.

---

*Final Year Project – Computer Science Department*  
*Real Estate Price Prediction System using Machine Learning*
