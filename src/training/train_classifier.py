"""
Train and compare rain-prediction classifiers (RandomForest, XGBoost).

Usage:
    python -m src.training.train_classifier
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data.cleaner import clean_weather_data
from src.data.loader import load_raw_data
from src.evaluation.metrics import classification_report_dict
from src.features.feature_engineering import build_features
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger
from src.utils.splitting import time_based_split

log = get_logger(__name__)


def get_feature_matrix(cfg):
    raw = load_raw_data(cfg)
    clean = clean_weather_data(raw)
    feats = build_features(clean, cfg["features"]["lags_hours"], cfg["features"]["rolling_windows_hours"])
    return feats


def train():
    cfg = load_config()
    log.info("Loading + cleaning + engineering features")
    df = get_feature_matrix(cfg)

    target = cfg["columns"]["target_rain"]
    drop_cols = ["timestamp", "location", "is_demo_data", target, "rainfall_mm", "source_weather_text", "wind_direction"]
    drop_cols += [c for c in df.columns if c.endswith("_is_outlier")]
    feature_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]

    train_df, val_df, test_df = time_based_split(
        df, cfg["training"]["test_size_days"], cfg["training"]["val_size_days"]
    )
    log.info(f"Train={len(train_df):,} Val={len(val_df):,} Test={len(test_df):,}")

    X_train, y_train = train_df[feature_cols], train_df[target]
    X_test, y_test = test_df[feature_cols], test_df[target]

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight="balanced",
            random_state=cfg["random_seed"], n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            eval_metric="logloss", random_state=cfg["random_seed"], n_jobs=-1,
        ),
    }

    results = {}
    best_name, best_model, best_f1 = None, None, -1
    for name, model in models.items():
        log.info(f"Training {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        report = classification_report_dict(y_test, y_pred, y_proba)
        results[name] = report
        log.info(f"{name}: {report}")
        if report["f1"] > best_f1:
            best_name, best_model, best_f1 = name, model, report["f1"]

    out_dir = resolve_path("models/classification")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, out_dir / "rain_classifier.joblib")
    joblib.dump(feature_cols, out_dir / "feature_columns.joblib")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2)

    log.info(f"Best model: {best_name} (F1={best_f1:.4f}) saved to {out_dir}")
    return results, best_name


if __name__ == "__main__":
    train()
