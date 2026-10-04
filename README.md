# 🌦️ AI Weather Prediction & Forecasting System

An end-to-end Machine Learning system that predicts weather (temperature, rain, humidity, wind speed, rainfall) from historical weather data and serves predictions via CLI and Streamlit dashboard.

---

## 1. Overview & Architecture

This repository contains a production-ready weather prediction platform with:
- **Data Collection:** Historical weather dataset (CSV) with automatic online fetching fallback via **Open-Meteo API** (or synthetic demo data generation if offline/missing).
- **Data Cleaning & Preprocessing:** Range validation, missing-value imputation (leakage-safe), IQR outlier flagging, and persistence to `data/processed/weather_clean.parquet`.
- **Feature Engineering:** Backward-only temporal features (month, day, hour, cyclical sin/cos), lag features (1h, 3h, 6h, 12h, 24h), and rolling statistics (3h, 6h, 12h, 24h).
- **Baseline Models:** Persistence forecast ($t = t-1$) and Linear/Logistic Regression baselines to establish benchmark metrics.
- **Main AI Models:**
  - **Random Forest** and **XGBoost** for Rain Classification (F1, Accuracy, Precision, Recall, ROC-AUC).
  - **Random Forest** and **XGBoost** for Regression targets (Temperature, Humidity, Wind Speed, Rainfall mm).
  - **LSTM Multi-Output Forecaster** (TensorFlow/Keras) for time-series sequence predictions.
- **Leakage-Safe Validation:** Time-based (chronological) train/validation/test splitting (no random shuffling).
- **Interfaces:** CLI tool and Streamlit interactive web application.

---

## 2. Model Performance Benchmarks (Real Historical Data)

### Classification (Rain Prediction)
| Model | F1 Score | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression (Baseline)** | 0.4058 | 0.6239 | 0.7368 | 0.2800 | 0.8915 |
| **XGBoost** | 0.5000 | 0.6881 | 0.9444 | 0.3400 | 0.7644 |
| **Random Forest (Best)** | **0.8190** | **0.8257** | 0.7818 | **0.8600** | **0.9129** |

### Regression Benchmarks (Next-Step Forecasting)

| Target Variable | Baseline (Persistence) MAE | Baseline (Linear Reg) MAE | Random Forest MAE | XGBoost MAE | Best Model |
|---|---|---|---|---|---|
| **Temperature (°C)** | 23.1899 | 19.9340 | 8.4666 | **4.4646** | **XGBoost** |
| **Humidity (%)** | 17.2202 | 6.5967 | **1.6321** | 1.6960 | **Random Forest** |
| **Wind Speed (km/h)** | 0.2752 | 3.1815 | **0.2087** | 0.6173 | **Random Forest** |
| **Rainfall (mm)** | 1.6275 | 2.0502 | **1.6477** | 1.7856 | **Random Forest** |

---

## 3. Installation & Setup

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables & Config (`.env`)
Copy `.env.example` to `.env` to configure optional API keys:
```bash
cp .env.example .env
```
*(Open-Meteo API works out-of-the-box without requiring an API key for standard usage).*

---

## 4. Usage Instructions

### Data Exploration (EDA)
Generate summary statistics and exploratory visual charts:
```bash
python -m scripts.explore_data
```
Outputs saved to `reports/figures/`:
- `eda_variable_distributions.png`
- `eda_monthly_trends.png`
- `eda_correlation_heatmap.png`

### Training Pipeline
Train rain classifier and regressors (including baselines):
```bash
python run_pipeline.py
```
To also train the **LSTM deep learning sequence model**:
```bash
python run_pipeline.py --lstm
```

### Making Predictions via CLI
Forecast weather for any city and optional target date:
```bash
# Default city (Toronto)
python -m src.prediction.predictor Toronto

# Specific city and date
python -m src.prediction.predictor Sydney --date 2024-01-15
```

### Running the Streamlit Web Application
Launch the interactive web dashboard:
```bash
streamlit run dashboard/app.py
```
Dashboard features:
1. **🏠 Home:** Summary stats & latest observations.
2. **🌦️ Prediction:** City & date interactive forecast query.
3. **📅 Forecast:** Multi-day historical trends.
4. **📊 Analytics:** Interactive distributions, seasonal trends & correlation heatmaps.
5. **🤖 Model Performance:** Detailed baseline vs ML metrics.
6. **🌍 Map:** Global observation map & yearly temperature race.

---

## 5. Running Tests

Run the full pytest suite:
```bash
python -m pytest tests/ -v
```

---

## 6. License
MIT License. See `LICENSE` for details.
