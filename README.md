# 🌦️ AI Weather Prediction & Forecasting System

A working, end-to-end ML system that trains rain / temperature / humidity /
wind-speed models on historical weather data and serves predictions through
a Streamlit dashboard.

> **This repo now ships with a REAL dataset already in `data/raw/weather_data.csv`**
> — 8,784 hourly observations for Toronto, 2012, from Environment Canada
> (sourced via a public GitHub mirror of the well-known "weather_2012"
> dataset). Full provenance of every column is documented in
> `scripts/prepare_real_dataset.py` and `data/README.md`. If you delete
> that file, the pipeline automatically falls back to synthetic demo data
> (clearly flagged `is_demo_data=True`) so it never silently breaks.
>
> **One honesty caveat:** the source data has no real precipitation-amount
> field, so `rainfall_mm` is 0.0 for every row — that regressor is skipped
> during training rather than faked (see section 11).

## 1. What's actually built here

This is a deliberately **leaner, real, runnable** version of a full
production weather-ML platform: 2 well-chosen models per task (RandomForest
+ XGBoost) instead of 6, one LSTM instead of three deep architectures, no
Docker/K8s ceremony. Every file contains working code — nothing is a stub.
If you want the extra model families (LightGBM, CatBoost, GRU/Transformer)
or the full notebook set, they follow the exact same pattern used in
`src/training/` and are straightforward to bolt on.

## 2. Features

- Synthetic demo-data generator with realistic diurnal/seasonal patterns (falls back automatically if no real CSV is present)
- Robust cleaning: timestamp parsing, dedup, physical-range validation, leakage-safe imputation, IQR outlier flagging (not blind removal)
- Leakage-safe feature engineering: temporal, lag (1/3/6/12/24h), rolling (3/6/12/24h) features, all computed backward-only and unit-tested for leakage
- Rain classifier (RandomForest vs XGBoost, best model auto-selected by F1)
- Regressors for temperature / humidity / wind_speed / rainfall_mm (RF vs XGBoost, best by MAE)
- LSTM multi-output sequence forecaster (TensorFlow/Keras)
- SHAP explainability for the rain classifier
- Time-based (chronological) train/val/test split — never random-shuffled
- Streamlit dashboard: Home, Prediction, Forecast trend, Analytics, Model Performance, Map
- Config-driven (`config/config.yaml`), logged, tested (`pytest`)

## 3. Why TensorFlow for the LSTM

Chosen over PyTorch because Keras's high-level `Model`/`fit` API keeps the
sequence-forecasting code short and readable, and `ModelCheckpoint` /
`EarlyStopping` callbacks map directly onto the "checkpointing + early
stopping" requirement with minimal boilerplate.

## 4. Project structure

```
AI-Weather-Prediction/
├── data/{raw,processed,external}/   # put your real CSV in data/raw/
├── src/
│   ├── data/          # loader.py (+ demo generator), cleaner.py
│   ├── features/      # feature_engineering.py
│   ├── training/       # train_classifier.py, train_regressor.py, train_lstm.py
│   ├── evaluation/     # metrics.py, explain.py (SHAP)
│   ├── prediction/     # predictor.py
│   └── utils/          # config.py, logger.py, splitting.py
├── models/{classification,regression,deep_learning}/
├── dashboard/app.py    # Streamlit app
├── tests/               # pytest suite (leakage test included)
├── config/config.yaml
├── run_pipeline.py      # one-command trainer
└── requirements.txt
```

## 5. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 6. Dataset setup

A real dataset is already included at `data/raw/weather_data.csv`
(Toronto, 2012, hourly — see provenance notes above). To use a different
real dataset, overwrite that file with columns described in
`data/README.md`, or adapt `scripts/prepare_real_dataset.py` for your
source. If the file is missing entirely, the pipeline **automatically
generates a synthetic demo dataset** so it never breaks — every row is
tagged `is_demo_data=True` in that case.

## 7. Train

```bash
python run_pipeline.py            # rain classifier + 4 regressors
python run_pipeline.py --lstm     # also train the LSTM (slower)
python -m src.evaluation.explain  # SHAP plot for the rain classifier
```

Each script also runs standalone, e.g. `python -m src.training.train_classifier`.

## 8. Dashboard

```bash
streamlit run dashboard/app.py
```

## 9. Tests

```bash
pytest tests/ -v
```

## 10. Example prediction (actual output from this repo, real data)

```
python -m src.prediction.predictor Toronto

{'location': 'Toronto', 'as_of': '2012-12-31 23:00:00', 'is_demo_data': False,
 'rain_probability': 4.2, 'rain_predicted': False, 'temperature': 0.1,
 'humidity': 85.8, 'wind_speed': 26.6, 'rainfall_mm': 0.0}
```

`rainfall_mm` is always 0.0 in the current dataset — see the caveat above,
this is not a model error, the source data has no real precipitation field.

## 11. Actual measured performance (REAL Toronto 2012 hourly data, chronological 20-day test split)

| Task | Best model | Key metric |
|---|---|---|
| Rain classification | XGBoost | F1 = 0.777, ROC-AUC = 0.982, Accuracy = 0.951 |
| Temperature regression (next hour) | XGBoost | MAE = 0.24 °C, R² = 0.996 |
| Humidity regression (next hour) | XGBoost | MAE = 0.30%, R² = 0.999 |
| Wind speed regression (next hour) | RandomForest | MAE = 3.15 km/h, R² = 0.786 |
| Rainfall regression | — | **skipped** — source has no real precipitation-amount field (constant 0.0 target) |

These are genuinely meaningful numbers from real weather physics (strong
autocorrelation hour-to-hour explains the high temperature/humidity R²;
wind speed is noisier and harder to predict, which shows up correctly as a
lower R²). Re-running `python run_pipeline.py` reproduces these exactly
(fixed `random_seed: 42`).

## 12. Explainability

`python -m src.evaluation.explain` saves a SHAP summary plot to
`reports/figures/shap_summary_rain_classifier.png` showing which features
(e.g. cloud cover, humidity, pressure) drive the rain prediction.

## 13. Known limitations

- Bundled real dataset covers **one city (Toronto) and one year (2012)** —
  good enough to prove the pipeline on real physics, not enough for
  robust production forecasting or multi-city comparison
- The dataset has **no real rainfall-amount field**, so `rainfall_mm`
  prediction is skipped entirely rather than trained on fake zeros
- `cloud_cover` in the bundled dataset is a coarse proxy derived from a
  text weather description, not a real sensor reading (see
  `scripts/prepare_real_dataset.py`)
- Single-point `predict_location_latest` only forecasts from the most
  recent row in history — true multi-horizon forecasting needs the LSTM
- Only 2 model families per task (not the full 6 originally requested) to
  keep this runnable and auditable rather than sprawling
- No Docker/CI included in this pass
- If you delete `data/raw/weather_data.csv`, predictions silently switch to
  synthetic demo data (clearly flagged) — don't mix the two in one comparison

## 14. Extending this project

- Add LightGBM/CatBoost by copying the `models = {...}` dict pattern in `train_classifier.py` / `train_regressor.py`
- Add GRU or a small Transformer by swapping the `build_model()` layers in `train_lstm.py`
- Add a `notebooks/` EDA notebook using the same `src.data` / `src.features` calls

## 15. License

MIT — see `LICENSE`.

---
*This is a research/educational forecasting system, not a replacement for official meteorological warnings.*
