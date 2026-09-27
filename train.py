"""
Life Expectancy Prediction Model Training Pipeline
Dataset: WHO Global Health Observatory (Kaggle: kumarajarshi/life-expectancy-who)
Trains Random Forest, Gradient Boosting, and Ridge Regressors.
Saves models, cross-validation metrics, feature importances, and country profiles.
"""

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Define directories
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), 'model_artifacts')
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

CSV_PATH = os.path.join(os.path.dirname(__file__), 'Life_Expectancy_Data.csv')

# Feature definitions with human-friendly metadata, units, and categories
FEATURE_METADATA = {
    'status_developed': {
        'label': 'Country Status',
        'type': 'binary',
        'category': 'Socioeconomic',
        'unit': 'Developed (1) or Developing (0)',
        'description': 'Development classification according to UN/WHO criteria'
    },
    'income_composition_of_resources': {
        'label': 'Income Composition / HDI',
        'type': 'continuous',
        'category': 'Socioeconomic',
        'unit': 'Index (0 to 1)',
        'description': 'Human Development Index in terms of income composition of resources'
    },
    'schooling': {
        'label': 'Average Schooling',
        'type': 'continuous',
        'category': 'Socioeconomic',
        'unit': 'Years',
        'description': 'Number of years of schooling'
    },
    'gdp': {
        'label': 'GDP per Capita',
        'type': 'continuous',
        'category': 'Socioeconomic',
        'unit': 'USD ($)',
        'description': 'Gross Domestic Product per capita in US Dollars'
    },
    'percentage_expenditure': {
        'label': 'Health Expenditure (% GDP)',
        'type': 'continuous',
        'category': 'Socioeconomic',
        'unit': '% of GDP',
        'description': 'Expenditure on health as a percentage of GDP per capita'
    },
    'total_expenditure': {
        'label': 'Govt Health Spending Ratio',
        'type': 'continuous',
        'category': 'Socioeconomic',
        'unit': '% of Budget',
        'description': 'General government expenditure on health as percentage of total government expenditure'
    },
    'population': {
        'label': 'Total Population',
        'type': 'continuous',
        'category': 'Socioeconomic',
        'unit': 'Inhabitants',
        'description': 'Population of the nation'
    },
    'polio': {
        'label': 'Polio Immunization Rate',
        'type': 'continuous',
        'category': 'Immunization & Health',
        'unit': '% Coverage',
        'description': 'Polio (Pol3) immunization coverage among 1-year-olds'
    },
    'diphtheria': {
        'label': 'Diphtheria Immunization Rate',
        'type': 'continuous',
        'category': 'Immunization & Health',
        'unit': '% Coverage',
        'description': 'DTP3 immunization coverage among 1-year-olds'
    },
    'hepatitis_b': {
        'label': 'Hepatitis B Immunization Rate',
        'type': 'continuous',
        'category': 'Immunization & Health',
        'unit': '% Coverage',
        'description': 'Hepatitis B (HepB) immunization coverage among 1-year-olds'
    },
    'hiv_aids': {
        'label': 'HIV/AIDS Mortality',
        'type': 'continuous',
        'category': 'Mortality & Diseases',
        'unit': 'Deaths / 1k live births',
        'description': 'Deaths per 1000 live births HIV/AIDS (ages 0-4)'
    },
    'adult_mortality': {
        'label': 'Adult Mortality Rate',
        'type': 'continuous',
        'category': 'Mortality & Diseases',
        'unit': 'Deaths / 1k adults',
        'description': 'Probability of dying between 15 and 60 years per 1000 population'
    },
    'infant_deaths': {
        'label': 'Infant Deaths',
        'type': 'continuous',
        'category': 'Mortality & Diseases',
        'unit': 'Deaths / 1k population',
        'description': 'Number of infant deaths per 1000 population'
    },
    'under_five_deaths': {
        'label': 'Under-Five Mortality',
        'type': 'continuous',
        'category': 'Mortality & Diseases',
        'unit': 'Deaths / 1k population',
        'description': 'Number of deaths of children under 5 years per 1000 population'
    },
    'measles': {
        'label': 'Measles Incidence',
        'type': 'continuous',
        'category': 'Mortality & Diseases',
        'unit': 'Reported cases',
        'description': 'Number of reported measles cases per 1000 population'
    },
    'bmi': {
        'label': 'Average BMI',
        'type': 'continuous',
        'category': 'Lifestyle & Nutrition',
        'unit': 'kg/m²',
        'description': 'Average Body Mass Index of the entire population'
    },
    'alcohol': {
        'label': 'Alcohol Consumption',
        'type': 'continuous',
        'category': 'Lifestyle & Nutrition',
        'unit': 'Litres pure alcohol/year',
        'description': 'Recorded per capita (15+) consumption in litres'
    },
    'thinness_1_19_years': {
        'label': 'Thinness (10-19 years)',
        'type': 'continuous',
        'category': 'Lifestyle & Nutrition',
        'unit': '% Prevalence',
        'description': 'Prevalence of thinness among children and adolescents'
    },
    'thinness_5_9_years': {
        'label': 'Thinness (5-9 years)',
        'type': 'continuous',
        'category': 'Lifestyle & Nutrition',
        'unit': '% Prevalence',
        'description': 'Prevalence of thinness among children aged 5 to 9'
    }
}

NUMERICAL_FEATURE_KEYS = [
    'adult_mortality', 'infant_deaths', 'alcohol', 'percentage_expenditure',
    'hepatitis_b', 'measles', 'bmi', 'under_five_deaths', 'polio',
    'total_expenditure', 'diphtheria', 'hiv_aids', 'gdp', 'population',
    'thinness_1_19_years', 'thinness_5_9_years', 'income_composition_of_resources', 'schooling'
]

ALL_FEATURE_KEYS = ['status_developed'] + NUMERICAL_FEATURE_KEYS

def clean_column_names(df):
    clean_map = {}
    for col in df.columns:
        clean = re.sub(r'\s+', '_', col.strip().lower())
        clean = clean.replace('/', '_').replace('-', '_')
        clean_map[col] = clean
    return df.rename(columns=clean_map)

def load_and_preprocess():
    print(f"Loading dataset from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    df = clean_column_names(df)
    
    initial_rows = len(df)
    # Drop rows missing target
    df = df.dropna(subset=['life_expectancy']).copy()
    print(f"Loaded {len(df)} valid records (dropped {initial_rows - len(df)} missing target).")

    # Missing value imputation: by country median first, then global median
    imputation_medians = {}
    for col in NUMERICAL_FEATURE_KEYS:
        global_med = float(df[col].median())
        imputation_medians[col] = global_med
        df[col] = df.groupby('country')[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(global_med)

    df['status_developed'] = (df['status'].astype(str).str.strip().str.lower() == 'developed').astype(int)

    return df, imputation_medians

def extract_country_profiles(df):
    """
    Extract the most recent record (usually 2015) for each country
    to serve as interactive preset profiles in the UI.
    """
    latest_df = df.sort_values(['country', 'year']).groupby('country').last().reset_index()
    profiles = {}
    for _, row in latest_df.iterrows():
        c_name = row['country']
        profile = {
            'country': c_name,
            'year': int(row['year']),
            'status': 'Developed' if row['status_developed'] == 1 else 'Developing',
            'status_developed': int(row['status_developed']),
            'actual_life_expectancy': round(float(row['life_expectancy']), 1),
            'features': {}
        }
        for f in ALL_FEATURE_KEYS:
            profile['features'][f] = round(float(row[f]), 3)
        profiles[c_name] = profile
    return profiles

def train():
    df, imputation_medians = load_and_preprocess()

    # Calculate feature summary stats for UI sliders/bounds
    feature_stats = {}
    for col in ALL_FEATURE_KEYS:
        s = df[col]
        meta = FEATURE_METADATA.get(col, {})
        feature_stats[col] = {
            'label': meta.get('label', col),
            'category': meta.get('category', 'General'),
            'unit': meta.get('unit', ''),
            'description': meta.get('description', ''),
            'min': round(float(s.min()), 2),
            'max': round(float(s.max()), 2),
            'mean': round(float(s.mean()), 2),
            'median': round(float(s.median()), 2),
            'q25': round(float(s.quantile(0.25)), 2),
            'q75': round(float(s.quantile(0.75)), 2),
            'std': round(float(s.std()), 2)
        }

    X = df[ALL_FEATURE_KEYS]
    y = df['life_expectancy']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples.")

    # Model training dictionary
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=250, max_depth=20, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, max_depth=5, random_state=42),
        'Ridge': Ridge(alpha=1.0)
    }

    metrics = {}
    trained_instances = {}

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        trained_instances[name] = model

        # Test evaluation
        y_pred = model.predict(X_test)
        r2 = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))

        # 5-fold cross validation on train
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')

        metrics[name] = {
            'test_r2': round(r2, 4),
            'test_rmse': round(rmse, 4),
            'test_mae': round(mae, 4),
            'cv_r2_mean': round(float(cv_scores.mean()), 4),
            'cv_r2_std': round(float(cv_scores.std()), 4)
        }
        print(f"Results for {name}:")
        print(f"  Test R²:  {r2:.4f}")
        print(f"  Test RMSE: {rmse:.4f} years")
        print(f"  Test MAE:  {mae:.4f} years")
        print(f"  CV 5-Fold R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Feature Importance from best model (Random Forest)
    best_model = trained_instances['Random Forest']
    rf_importances = best_model.feature_importances_
    importance_list = []
    for f, imp in sorted(zip(ALL_FEATURE_KEYS, rf_importances), key=lambda x: x[1], reverse=True):
        importance_list.append({
            'feature': f,
            'label': FEATURE_METADATA.get(f, {}).get('label', f),
            'category': FEATURE_METADATA.get(f, {}).get('category', 'General'),
            'importance': round(float(imp), 5),
            'percentage': round(float(imp) * 100, 2)
        })

    # Save models
    joblib.dump(best_model, os.path.join(ARTIFACTS_DIR, 'best_model.joblib'))
    joblib.dump(trained_instances['Gradient Boosting'], os.path.join(ARTIFACTS_DIR, 'gbm_model.joblib'))
    joblib.dump(trained_instances['Ridge'], os.path.join(ARTIFACTS_DIR, 'ridge_model.joblib'))
    print("Models saved successfully to model_artifacts/")

    # Extract country profiles
    country_profiles = extract_country_profiles(df)
    with open(os.path.join(ARTIFACTS_DIR, 'country_profiles.json'), 'w') as f:
        json.dump(country_profiles, f, indent=2)
    print(f"Country profiles saved for {len(country_profiles)} countries.")

    # Target statistics
    target_stats = {
        'count': int(len(df)),
        'mean': round(float(y.mean()), 2),
        'median': round(float(y.median()), 2),
        'min': round(float(y.min()), 2),
        'max': round(float(y.max()), 2),
        'std': round(float(y.std()), 2)
    }

    # Metadata package
    metadata = {
        'features': ALL_FEATURE_KEYS,
        'feature_metadata': FEATURE_METADATA,
        'feature_stats': feature_stats,
        'target_stats': target_stats,
        'metrics': metrics,
        'feature_importances': importance_list,
        'imputation_medians': imputation_medians,
        'countries': sorted(list(country_profiles.keys()))
    }

    with open(os.path.join(ARTIFACTS_DIR, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print("Metadata saved to model_artifacts/model_metadata.json")

    print("\nTraining and artifact generation complete!")

if __name__ == '__main__':
    train()
