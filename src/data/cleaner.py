"""
Data cleaning and validation.

Handles: missing values, duplicates, invalid timestamps, domain-range checks,
and outlier flagging (IQR-based) without blind removal. Also supports saving clean dataset to parquet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import load_config, resolve_path
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


def clean_weather_data(df: pd.DataFrame, save_processed: bool = False, cfg: dict | None = None) -> pd.DataFrame:
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
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            invalid = ~df[col].between(lo, hi) & df[col].notna()
            if invalid.sum():
                log.info(f"{col}: {invalid.sum()} values outside physical range [{lo},{hi}] -> set to NaN")
                df.loc[invalid, col] = np.nan

    # 4. Missing value imputation: forward-fill within each location's time series
    df = df.sort_values(["location", "timestamp"])
    numeric_cols = [c for c in VALID_RANGES if c in df.columns]
    for col in numeric_cols:
        df[col] = df.groupby("location")[col].ffill(limit=6)
        df[col] = df[col].fillna(df[col].median())

    # 5. Statistical outlier flagging (IQR)
    for col in numeric_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        df[f"{col}_is_outlier"] = ~df[col].between(lo, hi)

    log.info(f"Cleaning complete: {n_start:,} -> {len(df):,} rows")
    cleaned_df = df.reset_index(drop=True)

    if save_processed:
        cfg = cfg or load_config()
        processed_path = resolve_path(cfg["data"]["processed_path"])
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_parquet(processed_path, index=False)
        log.info(f"Saved processed clean dataset to {processed_path}")

    return cleaned_df


if __name__ == "__main__":
    from src.data.loader import load_raw_data
    from src.utils.config import load_config

    cfg = load_config()
    raw = load_raw_data(cfg)
    clean = clean_weather_data(raw, save_processed=True, cfg=cfg)
    print(clean.describe())
