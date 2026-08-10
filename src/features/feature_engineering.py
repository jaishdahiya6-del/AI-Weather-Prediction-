"""
Feature engineering: temporal, lag, and rolling features.

All lag/rolling features are computed per-location using only PAST values
(shift(1) before rolling), so no future information leaks into a row's
features. This is verified by test_no_leakage in tests/test_features.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE_NUMERIC = ["temperature", "humidity", "pressure", "wind_speed", "cloud_cover", "rainfall_mm"]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df["timestamp"]
    df["year"] = ts.dt.year
    df["month"] = ts.dt.month
    df["day"] = ts.dt.day
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["day_of_year"] = ts.dt.dayofyear
    df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["season"] = (df["month"] % 12 // 3).map({0: "winter", 1: "spring", 2: "summer", 3: "autumn"})
    # cyclical encodings so 23:00 and 00:00 are close in feature space
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_lag_features(df: pd.DataFrame, cols: list[str], lags: list[int]) -> pd.DataFrame:
    df = df.sort_values(["location", "timestamp"]).copy()
    for col in cols:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag_{lag}h"] = df.groupby("location")[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, cols: list[str], windows: list[int]) -> pd.DataFrame:
    df = df.sort_values(["location", "timestamp"]).copy()
    for col in cols:
        if col not in df.columns:
            continue
        # shift(1) first so the current row's own value is excluded from its rolling stats
        shifted = df.groupby("location")[col].shift(1)
        for w in windows:
            grp = shifted.groupby(df["location"])
            df[f"{col}_roll_mean_{w}h"] = grp.transform(lambda s: s.rolling(w, min_periods=1).mean())
            df[f"{col}_roll_std_{w}h"] = grp.transform(lambda s: s.rolling(w, min_periods=2).std())
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"temperature", "humidity"}.issubset(df.columns):
        df["temp_humidity_interaction"] = df["temperature"] * df["humidity"] / 100
    if {"temperature", "dew_point"}.issubset(df.columns):
        df["dew_point_depression"] = df["temperature"] - df["dew_point"]
    # IMPORTANT: a naive diff(1) on the raw column would be (current - previous),
    # which contains the current value itself -- a direct leak when `col` is also
    # the prediction target (current = previous + change_1h is trivially invertible).
    # Instead we compute the change between the two most recent PAST observations
    # (shift(1) - shift(2)), so this feature reflects "how fast was it changing
    # coming into this point" without containing the current-row value.
    for col in ["temperature", "pressure", "humidity", "wind_speed"]:
        if col in df.columns:
            lag1 = df.groupby("location")[col].shift(1)
            lag2 = df.groupby("location")[col].shift(2)
            df[f"{col}_change_1h"] = lag1 - lag2
    return df


def build_features(df: pd.DataFrame, lags: list[int], windows: list[int]) -> pd.DataFrame:
    """Full feature pipeline. Rows with NaN lag/rolling features (start of each
    location's series) are dropped since they cannot be used for supervised training."""
    df = add_temporal_features(df)
    df = add_lag_features(df, BASE_NUMERIC, lags)
    df = add_rolling_features(df, BASE_NUMERIC, windows)
    df = add_derived_features(df)
    df = pd.get_dummies(df, columns=["season"], prefix="season")

    feature_cols = [c for c in df.columns if any(k in c for k in ["_lag_", "_roll_", "_change_"])]
    df = df.dropna(subset=feature_cols)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    from src.data.cleaner import clean_weather_data
    from src.data.loader import load_raw_data
    from src.utils.config import load_config

    cfg = load_config()
    raw = load_raw_data(cfg)
    clean = clean_weather_data(raw)
    feats = build_features(clean, cfg["features"]["lags_hours"], cfg["features"]["rolling_windows_hours"])
    print(f"Feature matrix shape: {feats.shape}")
    print(feats.columns.tolist())
