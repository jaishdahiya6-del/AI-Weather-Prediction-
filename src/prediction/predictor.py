"""
Load trained models and produce predictions for a single input observation.

Because the trained models need lag/rolling history to build their features,
single-point prediction here uses the most recent historical rows for the
requested location (if available) to construct those features. If a model
file is missing, a clear error is raised rather than a silent fallback.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.data.cleaner import clean_weather_data
from src.data.loader import load_raw_data
from src.features.feature_engineering import build_features
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

log = get_logger(__name__)


class WeatherPredictor:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.clf_dir = resolve_path("models/classification")
        self.reg_dir = resolve_path("models/regression")
        self._history_cache: pd.DataFrame | None = None

    def _models_ready(self) -> bool:
        return (self.clf_dir / "rain_classifier.joblib").exists()

    def _get_history(self) -> pd.DataFrame:
        if self._history_cache is None:
            raw = load_raw_data(self.cfg)
            clean = clean_weather_data(raw)
            self._history_cache = build_features(
                clean, self.cfg["features"]["lags_hours"], self.cfg["features"]["rolling_windows_hours"]
            )
        return self._history_cache

    def predict_location_latest(self, location: str) -> dict:
        """Predict rain probability + regression targets using the most recent
        feature row available for `location` in the historical/demo dataset."""
        if not self._models_ready():
            raise FileNotFoundError(
                "No trained models found. Run `python -m src.training.train_classifier` "
                "and `python -m src.training.train_regressor` first."
            )

        history = self._get_history()
        loc_rows = history[history["location"] == location]
        if loc_rows.empty:
            raise ValueError(f"No historical data available for location '{location}'")
        latest = loc_rows.sort_values("timestamp").iloc[[-1]]

        result = {
            "location": location,
            "as_of": str(latest["timestamp"].iloc[0]),
            "is_demo_data": bool(latest["is_demo_data"].iloc[0]),
        }

        clf = joblib.load(self.clf_dir / "rain_classifier.joblib")
        clf_features = joblib.load(self.clf_dir / "feature_columns.joblib")
        X_clf = latest.reindex(columns=clf_features, fill_value=0)
        result["rain_probability"] = round(float(clf.predict_proba(X_clf)[0, 1]) * 100, 1)
        result["rain_predicted"] = bool(clf.predict(X_clf)[0])

        for target in ["temperature", "humidity", "wind_speed", "rainfall_mm"]:
            model_path = self.reg_dir / f"{target}_regressor.joblib"
            if not model_path.exists():
                continue
            reg = joblib.load(model_path)
            reg_features = joblib.load(self.reg_dir / f"{target}_feature_columns.joblib")
            X_reg = latest.reindex(columns=reg_features, fill_value=0)
            result[target] = round(float(reg.predict(X_reg)[0]), 1)

        return result


if __name__ == "__main__":
    import sys

    predictor = WeatherPredictor()
    location = sys.argv[1] if len(sys.argv) > 1 else "Toronto"
    try:
        print(predictor.predict_location_latest(location))
    except (FileNotFoundError, ValueError) as e:
        print(f"[Info] {e}")
