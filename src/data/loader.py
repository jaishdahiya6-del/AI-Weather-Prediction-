"""
Data loading.

If the real dataset is not found at config['data']['raw_path'], this module
generates a SYNTHETIC DEMO DATASET built from realistic statistical patterns
(diurnal temperature cycles, seasonal rainfall, correlated humidity/pressure).

This demo data is NOT real weather data. It exists only so the full pipeline
is runnable end-to-end before you plug in a real dataset (NOAA / Open-Meteo /
Kaggle / ERA5). Every row generated this way is tagged is_demo_data=True.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

log = get_logger(__name__)


def _generate_demo_data(cfg: dict) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["random_seed"])
    n_rows = cfg["data"]["demo_rows"]
    locations = cfg["data"]["locations"]

    start = pd.Timestamp("2018-01-01")
    hours = np.arange(n_rows)
    rows_per_location = n_rows // len(locations)

    frames = []
    for loc_idx, loc in enumerate(locations):
        n = rows_per_location
        ts = pd.date_range(start, periods=n, freq="h")
        doy = ts.dayofyear.values
        hour_of_day = ts.hour.values

        # seasonal + diurnal temperature signal + noise, offset per location
        base_temp = 15 + 10 * np.sin(2 * np.pi * (doy - 80) / 365) + loc_idx * 2
        diurnal = 5 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        temperature = base_temp + diurnal + rng.normal(0, 1.5, n)

        humidity = np.clip(70 - 0.8 * diurnal + rng.normal(0, 8, n), 5, 100)
        pressure = 1013 + rng.normal(0, 5, n) - 0.1 * diurnal
        wind_speed = np.clip(rng.gamma(2, 3, n), 0, 80)
        cloud_cover = np.clip(rng.beta(2, 2, n) * 100, 0, 100)
        visibility = np.clip(20 - cloud_cover / 10 + rng.normal(0, 2, n), 0.5, 20)
        dew_point = temperature - (100 - humidity) / 5

        rain_prob = np.clip((cloud_cover / 100) * 0.6 + (humidity / 100) * 0.4 - 0.25, 0, 1)
        rain = rng.binomial(1, rain_prob)
        rainfall_mm = np.where(rain == 1, rng.gamma(2, 3, n), 0.0)

        frames.append(pd.DataFrame({
            "timestamp": ts,
            "location": loc,
            "latitude": 20 + loc_idx * 3.0,
            "longitude": 70 + loc_idx * 4.0,
            "temperature": temperature.round(1),
            "humidity": humidity.round(1),
            "pressure": pressure.round(1),
            "wind_speed": wind_speed.round(1),
            "wind_direction": rng.integers(0, 360, n),
            "cloud_cover": cloud_cover.round(1),
            "visibility": visibility.round(1),
            "dew_point": dew_point.round(1),
            "rain": rain,
            "rainfall_mm": rainfall_mm.round(1),
            "is_demo_data": True,
        }))

    df = pd.concat(frames, ignore_index=True)
    log.info(f"Generated SYNTHETIC DEMO dataset: {len(df):,} rows across {len(locations)} locations")
    return df


def load_raw_data(cfg: dict | None = None) -> pd.DataFrame:
    """Load the raw dataset. Falls back to synthetic demo data if not found."""
    cfg = cfg or load_config()
    raw_path = resolve_path(cfg["data"]["raw_path"])

    if raw_path.exists():
        log.info(f"Loading real dataset from {raw_path}")
        df = pd.read_csv(raw_path)
        df["is_demo_data"] = False
        return df

    if not cfg["data"].get("demo_mode", True):
        raise FileNotFoundError(
            f"Dataset not found at {raw_path} and demo_mode is disabled. "
            "Place a CSV file with weather observations there, or enable demo_mode in config.yaml."
        )

    log.warning(
        f"No dataset found at {raw_path}. Falling back to DEMO MODE. "
        "Place your real CSV there (see data/README.md) for real predictions."
    )
    return _generate_demo_data(cfg)


if __name__ == "__main__":
    config = load_config()
    data = load_raw_data(config)
    print(data.head())
    print(f"\nShape: {data.shape}")
    print(f"Demo data: {data['is_demo_data'].iloc[0]}")
