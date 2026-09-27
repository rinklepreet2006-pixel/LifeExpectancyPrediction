"""
Flask Web Application for WHO Life Expectancy Prediction
Provides REST APIs for real-time predictions, feature importance,
country profiles, and dataset exploration.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import re
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'model_artifacts')
CSV_PATH = os.path.join(BASE_DIR, 'Life_Expectancy_Data.csv')

# Load models and metadata on startup
models = {
    'Random Forest': joblib.load(os.path.join(ARTIFACTS_DIR, 'best_model.joblib')),
    'Gradient Boosting': joblib.load(os.path.join(ARTIFACTS_DIR, 'gbm_model.joblib')),
    'Ridge': joblib.load(os.path.join(ARTIFACTS_DIR, 'ridge_model.joblib'))
}

with open(os.path.join(ARTIFACTS_DIR, 'model_metadata.json'), 'r') as f:
    METADATA = json.load(f)

with open(os.path.join(ARTIFACTS_DIR, 'country_profiles.json'), 'r') as f:
    COUNTRY_PROFILES = json.load(f)

# Load raw dataset for sample exploration
def get_clean_dataset():
    df = pd.read_csv(CSV_PATH)
    clean_map = {}
    for col in df.columns:
        c = re.sub(r'\s+', '_', col.strip().lower()).replace('/', '_').replace('-', '_')
        clean_map[col] = c
    df = df.rename(columns=clean_map)
    df = df.dropna(subset=['life_expectancy'])
    return df

DATASET_DF = get_clean_dataset()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    """Return model metrics, feature metadata, statistics, and country list."""
    return jsonify({
        'status': 'success',
        'data': METADATA
    })

@app.route('/api/countries', methods=['GET'])
def get_countries():
    """Return list of available countries."""
    countries = sorted(list(COUNTRY_PROFILES.keys()))
    return jsonify({
        'status': 'success',
        'count': len(countries),
        'countries': countries
    })

@app.route('/api/country/<country_name>', methods=['GET'])
def get_country_profile(country_name):
    """Return the profile for a given country."""
    profile = COUNTRY_PROFILES.get(country_name)
    if not profile:
        # Case insensitive match fallback
        for k, v in COUNTRY_PROFILES.items():
            if k.lower() == country_name.lower():
                profile = v
                break
    
    if not profile:
        return jsonify({'status': 'error', 'message': f'Country "{country_name}" not found.'}), 404

    return jsonify({
        'status': 'success',
        'data': profile
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Accepts feature JSON and model choice, returns predicted life expectancy,
    category, comparison to global average, and feature impact analysis.
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'status': 'error', 'message': 'Invalid JSON body'}), 400

        selected_model_name = data.get('model', 'Random Forest')
        if selected_model_name not in models:
            selected_model_name = 'Random Forest'

        model = models[selected_model_name]
        feature_keys = METADATA['features']

        # Extract feature vector in correct order, using imputation median as default
        feature_vector = []
        raw_inputs = data.get('features', {})
        for key in feature_keys:
            val = raw_inputs.get(key)
            if val is None or val == '':
                val = METADATA['imputation_medians'].get(key, 0.0)
            else:
                try:
                    val = float(val)
                except ValueError:
                    val = METADATA['imputation_medians'].get(key, 0.0)
            feature_vector.append(val)

        X_input = np.array([feature_vector])
        pred_value = float(model.predict(X_input)[0])
        # Bound realistically to 20-100 years
        pred_value = max(25.0, min(95.0, round(pred_value, 1)))

        # Also get predictions from the other models for comparison
        rf_pred = round(float(models['Random Forest'].predict(X_input)[0]), 1)
        gbm_pred = round(float(models['Gradient Boosting'].predict(X_input)[0]), 1)
        ridge_pred = round(float(models['Ridge'].predict(X_input)[0]), 1)

        # Global average comparison
        global_avg = METADATA['target_stats']['mean']  # ~69.2
        delta = round(pred_value - global_avg, 1)

        # Longevity Tier categorization
        if pred_value >= 80.0:
            tier = {'name': 'Exceptional Longevity', 'color': '#10b981', 'badge': 'tier-exceptional', 'rank': 'Top 15% Globally'}
        elif pred_value >= 74.0:
            tier = {'name': 'High Longevity', 'color': '#06b6d4', 'badge': 'tier-high', 'rank': 'Above Global Average'}
        elif pred_value >= 66.0:
            tier = {'name': 'Moderate / Developing Longevity', 'color': '#f59e0b', 'badge': 'tier-moderate', 'rank': 'Near Global Average'}
        else:
            tier = {'name': 'Critical Health Vulnerability', 'color': '#ef4444', 'badge': 'tier-critical', 'rank': 'Below Global Average'}

        # Calculate feature impact breakdown relative to dataset medians
        impacts = []
        # Higher values of these features are generally negative for life expectancy
        negative_factor_keys = {'hiv_aids', 'adult_mortality', 'infant_deaths', 'under_five_deaths', 'measles', 'thinness_1_19_years', 'thinness_5_9_years'}

        for key, val in zip(feature_keys, feature_vector):
            stats = METADATA['feature_stats'].get(key)
            if not stats:
                continue
            median = stats['median']
            diff = val - median
            label = stats['label']
            is_negative_factor = key in negative_factor_keys

            # Determine whether this feature contributes positively or negatively to longevity
            if is_negative_factor:
                # e.g., low HIV is good (+ impact), high HIV is bad (- impact)
                impact_direction = 'positive' if diff < 0 else ('negative' if diff > 0 else 'neutral')
            else:
                # e.g., high schooling is good (+ impact), low schooling is bad (- impact)
                impact_direction = 'positive' if diff > 0 else ('negative' if diff < 0 else 'neutral')

            # Normalized relative deviation
            std = stats['std'] if stats['std'] > 0 else 1.0
            z_score = diff / std

            # Weight by feature importance
            importance_weight = next((item['importance'] for item in METADATA['feature_importances'] if item['feature'] == key), 0.01)
            score = abs(z_score) * importance_weight

            impacts.append({
                'feature': key,
                'label': label,
                'category': stats['category'],
                'user_val': val,
                'median_val': median,
                'unit': stats['unit'],
                'direction': impact_direction,
                'score': round(float(score), 4)
            })

        # Sort top positive and top negative contributors
        positive_factors = sorted([x for x in impacts if x['direction'] == 'positive'], key=lambda x: x['score'], reverse=True)[:3]
        negative_factors = sorted([x for x in impacts if x['direction'] == 'negative'], key=lambda x: x['score'], reverse=True)[:3]

        return jsonify({
            'status': 'success',
            'prediction': pred_value,
            'model_used': selected_model_name,
            'benchmark_comparison': {
                'global_avg': global_avg,
                'delta': delta,
                'direction': 'higher' if delta >= 0 else 'lower'
            },
            'tier': tier,
            'model_comparisons': {
                'Random Forest': rf_pred,
                'Gradient Boosting': gbm_pred,
                'Ridge': ridge_pred
            },
            'top_drivers': {
                'positive': positive_factors,
                'negative': negative_factors
            }
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/dataset/sample', methods=['GET'])
def get_dataset_sample():
    """Return paginated and searchable rows from the WHO dataset."""
    country_query = request.args.get('search', '').strip().lower()
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))

    filtered_df = DATASET_DF
    if country_query:
        filtered_df = filtered_df[filtered_df['country'].str.lower().str.contains(country_query)]

    total_records = len(filtered_df)
    start = (page - 1) * per_page
    end = start + per_page
    paged_df = filtered_df.iloc[start:end]

    cols_to_show = [
        'country', 'year', 'status', 'life_expectancy', 'adult_mortality',
        'infant_deaths', 'bmi', 'hiv_aids', 'gdp', 'schooling', 'income_composition_of_resources'
    ]

    records = paged_df[cols_to_show].to_dict(orient='records')
    # Round float values for clean presentation
    for r in records:
        for k, v in r.items():
            if isinstance(v, float):
                r[k] = round(v, 2)

    return jsonify({
        'status': 'success',
        'total': total_records,
        'page': page,
        'per_page': per_page,
        'total_pages': int(np.ceil(total_records / per_page)),
        'records': records
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Life Expectancy WHO Predictor Server on http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=False)
