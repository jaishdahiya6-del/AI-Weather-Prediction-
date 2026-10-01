"""
Load trained models and produce predictions for a city and target date/time.

If the requested city/date is not in local historical cache, attempts online fetch via
Open-Meteo API or predicts using the latest available features for that location.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from src.data.api_fetcher import fetch_open_meteo_historical
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

    def predict_location_date(self, location: str, target_date: Optional[str] = None) -> dict:
        """Predict weather for a location and optional target_date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)."""
        if not self._models_ready():
            raise FileNotFoundError(
                "No trained models found. Run `python run_pipeline.py` first."
            )

        history = self._get_history()
        loc_rows = history[history["location"].str.lower() == location.strip().lower()]

        row_to_use = None

        if not loc_rows.empty:
            if target_date:
                target_dt = pd.to_datetime(target_date)
                loc_rows_sorted = loc_rows.sort_values("timestamp")
                past_rows = loc_rows_sorted[loc_rows_sorted["timestamp"] <= target_dt]
                if not past_rows.empty:
                    row_to_use = past_rows.iloc[[-1]]
                else:
                    row_to_use = loc_rows_sorted.iloc[[0]]
            else:
                row_to_use = loc_rows.sort_values("timestamp").iloc[[-1]]

        # If location not in local dataset, try fetching recent historical data via Open-Meteo
        if row_to_use is None and target_date:
            try:
                dt = pd.to_datetime(target_date)
                start_str = (dt - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
                end_str = dt.strftime("%Y-%m-%d")
                df_api = fetch_open_meteo_historical(location, start_str, end_str)
                df_clean = clean_weather_data(df_api)
                df_feats = build_features(
                    df_clean, self.cfg["features"]["lags_hours"], self.cfg["features"]["rolling_windows_hours"]
                )
                if not df_feats.empty:
                    row_to_use = df_feats.sort_values("timestamp").iloc[[-1]]
            except Exception as e:
                log.warning(f"Failed to fetch Open-Meteo data for {location}: {e}")

        if row_to_use is None or row_to_use.empty:
            raise ValueError(f"No weather data found or fetchable for location '{location}' on/near {target_date}")

        result = {
            "location": str(row_to_use["location"].iloc[0]),
            "as_of": str(row_to_use["timestamp"].iloc[0]),
            "target_date_requested": target_date or "latest",
            "is_demo_data": bool(row_to_use["is_demo_data"].iloc[0]) if "is_demo_data" in row_to_use.columns else False,
        }

        clf = joblib.load(self.clf_dir / "rain_classifier.joblib")
        clf_features = joblib.load(self.clf_dir / "feature_columns.joblib")
        X_clf = row_to_use.reindex(columns=clf_features, fill_value=0)
        result["rain_probability"] = round(float(clf.predict_proba(X_clf)[0, 1]) * 100, 1)
        result["rain_predicted"] = bool(clf.predict(X_clf)[0])

        for target in ["temperature", "humidity", "wind_speed", "rainfall_mm"]:
            model_path = self.reg_dir / f"{target}_regressor.joblib"
            if not model_path.exists():
                continue
            reg = joblib.load(model_path)
            reg_features = joblib.load(self.reg_dir / f"{target}_feature_columns.joblib")
            X_reg = row_to_use.reindex(columns=reg_features, fill_value=0)
            result[target] = round(float(reg.predict(X_reg)[0]), 1)

        return result

    def predict_location_latest(self, location: str) -> dict:
        return self.predict_location_date(location, target_date=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Weather Predictor CLI")
    parser.add_argument("city", type=str, nargs="?", default="Toronto", help="City name (e.g. Toronto, Sydney, London)")
    parser.add_argument("--date", type=str, default=None, help="Target forecast date (YYYY-MM-DD)")
    args = parser.parse_args()

    predictor = WeatherPredictor()
    try:
        res = predictor.predict_location_date(args.city, args.date)
        print("\n================ Forecast Result ================")
        for k, v in res.items():
            print(f"  {k}: {v}")
        print("=================================================\n")
    except (FileNotFoundError, ValueError) as e:
        print(f"[Error] {e}")
