"""
Train and compare regressors (RandomForest, XGBoost) for multiple weather
targets: temperature, humidity, wind_speed, rainfall_mm.

Usage:
    python -m src.training.train_regressor
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.evaluation.metrics import regression_report_dict
from src.training.train_classifier import get_feature_matrix
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger
from src.utils.splitting import time_based_split

log = get_logger(__name__)

TARGETS = ["temperature", "humidity", "wind_speed", "rainfall_mm"]


def train():
    cfg = load_config()
    df = get_feature_matrix(cfg)

    train_df, val_df, test_df = time_based_split(
        df, cfg["training"]["test_size_days"], cfg["training"]["val_size_days"]
    )

    out_dir = resolve_path("models/regression")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for target in TARGETS:
        drop_cols = ["timestamp", "location", "is_demo_data", "rain", "source_weather_text", "wind_direction"] + TARGETS
        drop_cols += [c for c in df.columns if c.endswith("_is_outlier")]
        # dew_point is a deterministic function of same-timestamp temperature+humidity
        # (Magnus formula) -- keeping it as a raw feature would let the temperature/
        # humidity regressors invert that formula and "cheat" instead of learning
        # real patterns. Its lagged/rolling versions are fine (past values only).
        drop_cols += ["dew_point"]
        feature_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]

        X_train, y_train = train_df[feature_cols], train_df[target]
        X_test, y_test = test_df[feature_cols], test_df[target]

        models = {
            "random_forest": RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=cfg["random_seed"], n_jobs=-1
            ),
            "xgboost": XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                random_state=cfg["random_seed"], n_jobs=-1,
            ),
        }

        if y_train.nunique() <= 1:
            log.warning(
                f"[{target}] target has {y_train.nunique()} unique value(s) in the training "
                "set (constant) -- skipping training. Any R2=1.0 you might see for a constant "
                "target is a degenerate artifact, not real predictive skill."
            )
            all_results[target] = {
                "results": None,
                "best_model": None,
                "skipped_reason": "Target is constant in this dataset (see data/README.md / "
                                   "scripts/prepare_real_dataset.py for why).",
            }
            continue

        results = {}
        best_name, best_model, best_mae = None, None, float("inf")
        for name, model in models.items():
            log.info(f"[{target}] training {name}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report = regression_report_dict(y_test, y_pred)
            results[name] = report
            log.info(f"[{target}] {name}: {report}")
            if report["mae"] < best_mae:
                best_name, best_model, best_mae = name, model, report["mae"]

        joblib.dump(best_model, out_dir / f"{target}_regressor.joblib")
        joblib.dump(feature_cols, out_dir / f"{target}_feature_columns.joblib")
        all_results[target] = {"results": results, "best_model": best_name}
        log.info(f"[{target}] best: {best_name} (MAE={best_mae:.4f})")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


if __name__ == "__main__":
    train()
