"""
Flask web application – Real Estate Price Prediction System.
Serves the prediction form, handles API requests, and provides admin controls.
"""

import os
import json
import sys
from functools import wraps

from flask import (Flask, render_template, request, jsonify,
                   session, redirect, url_for, flash)

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    SECRET_KEY, ADMIN_PASSWORD, DEBUG, HOST, PORT,
    MODEL_PIPELINE_PATH, METRICS_PATH, NEIGHBOURHOODS, PROPERTY_TYPES
)
from src.prediction_engine import predict, get_recent_predictions, load_pipeline

app = Flask(__name__)
app.secret_key = SECRET_KEY


def _ensure_model_ready():
    """Auto-train model if pipeline file is missing."""
    if not os.path.exists(MODEL_PIPELINE_PATH):
        app.logger.warning("Model pipeline not found – running training pipeline...")
        from train import main as train_main
        train_main()


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated


# ─────────────────────────── Public routes ───────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict')
def predict_page():
    return render_template(
        'predict.html',
        neighbourhoods=NEIGHBOURHOODS,
        property_types=PROPERTY_TYPES,
    )


# ─────────────────────────── Prediction API ──────────────────────────

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'status': 'error', 'errors': ['No JSON payload received.']}), 400

        result = predict(data)

        if result.get('status') == 'error':
            return jsonify(result), 400

        return jsonify(result), 200

    except FileNotFoundError as e:
        return jsonify({'status': 'error', 'errors': [str(e)]}), 503
    except Exception as e:
        app.logger.exception("Unexpected error during prediction")
        return jsonify({'status': 'error', 'errors': ['Internal server error.']}), 500


# ─────────────────────────── Admin routes ────────────────────────────

@app.route('/admin', methods=['GET'])
@login_required
def admin():
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)

    recent_predictions = get_recent_predictions(20)
    return render_template('admin.html', metrics=metrics,
                           predictions=recent_predictions)


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        password = request.form.get('password', '')
        if password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            return redirect(url_for('admin'))
        flash('Incorrect password. Please try again.', 'danger')
    return render_template('admin_login.html')


@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))


@app.route('/admin/retrain', methods=['POST'])
@login_required
def admin_retrain():
    try:
        from train import main as train_main
        train_main()
        # Reload pipeline after retraining
        global _pipeline
        import src.prediction_engine as pe
        pe._pipeline = None
        flash('Model retrained successfully!', 'success')
    except Exception as e:
        flash(f'Retraining failed: {str(e)}', 'danger')
    return redirect(url_for('admin'))


@app.route('/admin/upload', methods=['POST'])
@login_required
def admin_upload():
    from config import RAW_DATA_PATH
    if 'dataset' not in request.files:
        flash('No file selected.', 'danger')
        return redirect(url_for('admin'))

    file = request.files['dataset']
    if not file.filename.endswith('.csv'):
        flash('Only CSV files are accepted.', 'danger')
        return redirect(url_for('admin'))

    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    file.save(RAW_DATA_PATH)
    flash('Dataset uploaded successfully. Click "Retrain Model" to update.', 'success')
    return redirect(url_for('admin'))


@app.route('/api/metrics')
def api_metrics():
    if not os.path.exists(METRICS_PATH):
        return jsonify({'error': 'Metrics not available'}), 404
    with open(METRICS_PATH) as f:
        return jsonify(json.load(f))


if __name__ == '__main__':
    _ensure_model_ready()
    app.run(host=HOST, port=PORT, debug=DEBUG)
