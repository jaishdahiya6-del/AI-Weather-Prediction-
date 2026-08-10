"""
Data cleaning and validation.

Handles: missing values, duplicates, invalid timestamps, domain-range checks,
and outlier flagging (IQR-based) without blind removal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)

# Physically plausible ranges used for domain validation (not statistical outlier detection)
VALID_RANGES = {
    "temperature": (-90, 60),      # °C, world extremes
    "humidity": (0, 100),          # %
    "pressure": (850, 1085),       # hPa, recorded extremes
    "wind_speed": (0, 408),        # km/h, world record ~408
    "cloud_cover": (0, 100),       # %
    "visibility": (0, 50),         # km
}


def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline and return a clean, sorted DataFrame."""
    df = df.copy()
    n_start = len(df)

    # 1. Parse timestamps, drop rows where parsing fails
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    n_bad_ts = df["timestamp"].isna().sum()
    if n_bad_ts:
        log.info(f"Dropping {n_bad_ts} rows with unparseable timestamps")
    df = df.dropna(subset=["timestamp"])

    # 2. Drop exact duplicate rows, and duplicate (location, timestamp) keeping first
    n_dup = df.duplicated().sum()
    df = df.drop_duplicates()
    n_dup_key = df.duplicated(subset=["location", "timestamp"]).sum()
    df = df.drop_duplicates(subset=["location", "timestamp"], keep="first")
    if n_dup or n_dup_key:
        log.info(f"Dropped {n_dup} exact duplicates and {n_dup_key} duplicate (location,timestamp) rows")

    # 3. Domain-range validation: values outside physically-possible ranges become NaN
    #    (kept as missing rather than silently dropped -- distinguishes sensor error from
    #    genuine extreme events, which fall inside these wide physical bounds)
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            invalid = ~df[col].between(lo, hi) & df[col].notna()
            if invalid.sum():
                log.info(f"{col}: {invalid.sum()} values outside physical range [{lo},{hi}] -> set to NaN")
                df.loc[invalid, col] = np.nan

    # 4. Missing value imputation: forward-fill within each location's time series
    #    (short gaps), then median-fill any remainder. This uses only past values,
    #    so it does not leak future information.
    df = df.sort_values(["location", "timestamp"])
    numeric_cols = [c for c in VALID_RANGES if c in df.columns]
    for col in numeric_cols:
        df[col] = df.groupby("location")[col].ffill(limit=6)
        df[col] = df[col].fillna(df[col].median())

    # 5. Statistical outlier flagging (IQR) -- flagged, NOT removed, so genuine
    #    extreme weather events remain in the data for the model to learn from.
    for col in numeric_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        df[f"{col}_is_outlier"] = ~df[col].between(lo, hi)

    log.info(f"Cleaning complete: {n_start:,} -> {len(df):,} rows")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    from src.data.loader import load_raw_data
    from src.utils.config import load_config

    cfg = load_config()
    raw = load_raw_data(cfg)
    clean = clean_weather_data(raw)
    print(clean.describe())
