# 🫀 VitaPulse • WHO Life Expectancy AI & Clinical Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Backend-Flask-black.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Accuracy](https://img.shields.io/badge/R%C2%B2_Score-96.64%25-brightgreen.svg)](#model-performance)
[![Dataset](https://img.shields.io/badge/Dataset-WHO_GHO_(Kaggle)-informational.svg)](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)

An end-to-end Machine Learning intelligence system and interactive web application that predicts human life expectancy across **193 nations** using historical World Health Organization (WHO) Global Health Observatory indicators (2000–2015).

---

## 🌟 Key Features

- **⚡ High-Accuracy AI Engine**:
  - **Random Forest Regressor** achieving **$R^2 = 96.64\%$** and a Mean Absolute Error (MAE) of only **$\pm 1.07$ years**.
  - Compared side-by-side with **Gradient Boosting Regressor** ($R^2 = 96.41\%$) and regularized **Ridge Regression** ($R^2 = 81.95\%$).
- **🎨 Glassmorphic Web Dashboard**:
  - Deep slate & midnight navy palette with glowing health indicators.
  - Interactive SVG radial gauge displaying real-time longevity predictions.
  - Longevity tier classification (Exceptional, High, Moderate, Critical Vulnerability).
  - Cross-model consensus view for multi-algorithm validation.
- **🌍 Country Presets & "What-If" Longevity Simulator**:
  - One-click profile loader for 183 countries (e.g. Japan, Switzerland, USA, Germany, India, Brazil, Nigeria, Kenya).
  - 19 interactive parameters categorized into **Socioeconomic**, **Mortality & Disease**, **Immunization**, and **Lifestyle & Nutrition**.
  - Real-time debounced auto-predict mode as sliders move.
- **🔍 Explainable AI (XAI)**:
  - Dynamic breakdown of individual protective factors (e.g., high schooling, universal immunization) and mortality risk factors.
  - Gini feature importance analysis highlighting the dominant influence of pediatric HIV/AIDS control, Human Development Index (HDI), and adult mortality.
- **🌐 Interactive WHO Dataset Explorer**:
  - Searchable, paginated table of the 2,928 WHO records.
  - "Inspect" button to load any historical nation-year directly into the simulator.

---

## 📊 Model Performance & Benchmarks

All models were evaluated on an 80/20 train/test split with 5-fold cross-validation:

| Model | Test $R^2$ Score | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | 5-Fold Cross Validation $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| 🌲 **Random Forest** | **0.9664 (96.6%)** | **1.0700 yrs** | **1.7059 yrs** | **0.9570 $\pm$ 0.0039** |
| ⚡ **Gradient Boosting** | 0.9641 (96.4%) | 1.1887 yrs | 1.7622 yrs | 0.9579 $\pm$ 0.0036 |
| 📈 **Ridge Linear** | 0.8195 (82.0%) | 2.9283 yrs | 3.9508 yrs | 0.8143 $\pm$ 0.0135 |

### Top Predictive Drivers
1. **HIV/AIDS Mortality (Ages 0-4)**: ~59.7% weight
2. **Income Composition of Resources (HDI)**: ~17.1% weight
3. **Adult Mortality Rate**: ~12.7% weight
4. **Body Mass Index (BMI)**: ~2.0% weight
5. **Under-Five Deaths**: ~1.6% weight
6. **Average Schooling**: ~1.1% weight

---

## 📂 Project Architecture

```
iml project/
├── Life_Expectancy_Data.csv    # WHO Global Health Observatory dataset
├── train.py                    # Model training, cross-validation & artifact exporter
├── app.py                      # Flask REST API & Web Server
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── README.md                   # Documentation
├── model_artifacts/
│   ├── best_model.joblib       # Pre-trained Random Forest model
│   ├── gbm_model.joblib        # Pre-trained Gradient Boosting model
│   ├── ridge_model.joblib      # Pre-trained Ridge Regression model
│   ├── model_metadata.json     # Feature stats, metadata & benchmark results
│   └── country_profiles.json   # Latest WHO profiles for 183 nations
├── templates/
│   └── index.html              # HTML5 single-page application
└── static/
    ├── css/
    │   └── style.css           # Glassmorphism design system & animations
    └── js/
        └── app.js              # Real-time SVG gauge, sliders & API client
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/life-expectancy-prediction.git
cd life-expectancy-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Retrain Models
To run the training pipeline and generate fresh evaluation metrics:
```bash
python train.py
```

### 4. Run the Web Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🔌 API Endpoints

- `GET /api/metadata` — Model metrics, feature distributions, and ranges.
- `GET /api/countries` — List of available WHO nations.
- `GET /api/country/<country_name>` — Latest WHO recorded profile for the specified nation.
- `POST /api/predict` — Computes life expectancy prediction, longevity tier, delta comparison, and driver impacts for given features.
- `GET /api/dataset/sample?search=<query>&page=<num>` — Paginated search over the raw WHO dataset.

---

## 📜 Dataset Citation

- **Original Source**: World Health Organization (WHO) Global Health Observatory (GHO)
- **Kaggle Mirror**: [Kumar Rajarshi - Life Expectancy (WHO)](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)
