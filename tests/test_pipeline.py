import pandas as pd
import pytest

from src.data.api_fetcher import geocode_city
from src.data.cleaner import clean_weather_data
from src.features.feature_engineering import build_features
from src.models.baseline import evaluate_linear_regression, evaluate_persistence_regressor
from src.prediction.predictor import WeatherPredictor
from src.utils.config import load_config
from src.utils.splitting import time_based_split


@pytest.fixture(scope="module")
def raw_demo_df():
    from src.data.loader import _generate_demo_data
    cfg = load_config()
    cfg["data"]["demo_rows"] = 2000
    return _generate_demo_data(cfg)


def test_demo_data_generation(raw_demo_df):
    assert len(raw_demo_df) > 0
    assert raw_demo_df["is_demo_data"].all()
    assert {"temperature", "humidity", "rain", "rainfall_mm"}.issubset(raw_demo_df.columns)


def test_cleaning_removes_duplicates(raw_demo_df):
    dup = pd.concat([raw_demo_df, raw_demo_df.iloc[:5]], ignore_index=True)
    cleaned = clean_weather_data(dup)
    assert not cleaned.duplicated(subset=["location", "timestamp"]).any()


def test_cleaning_handles_impossible_values(raw_demo_df):
    df = raw_demo_df.copy()
    df.loc[0, "temperature"] = 999  # impossible
    cleaned = clean_weather_data(df)
    assert cleaned["temperature"].max() < 999


def test_no_missing_after_cleaning(raw_demo_df):
    cleaned = clean_weather_data(raw_demo_df)
    numeric_cols = ["temperature", "humidity", "pressure", "wind_speed"]
    assert cleaned[numeric_cols].isna().sum().sum() == 0


def test_feature_engineering_no_leakage(raw_demo_df):
    """A lag/rolling feature for row i must only use data with timestamp < row i's timestamp."""
    cleaned = clean_weather_data(raw_demo_df)
    feats = build_features(cleaned, lags=[1, 3], windows=[3])
    sample = feats[feats["location"] == feats["location"].iloc[0]].sort_values("timestamp")
    # lag_1h value at row i should equal temperature at row i-1 (within the raw cleaned series)
    merged = sample[["timestamp", "temperature_lag_1h"]].merge(
        cleaned[cleaned["location"] == sample["location"].iloc[0]][["timestamp", "temperature"]]
        .assign(timestamp=lambda d: d["timestamp"] + pd.Timedelta(hours=1))
        .rename(columns={"temperature": "expected_lag_1h"}),
        on="timestamp", how="left",
    )
    diffs = (merged["temperature_lag_1h"] - merged["expected_lag_1h"]).dropna().abs()
    assert (diffs < 1e-6).all()


def test_change_feature_excludes_current_row(raw_demo_df):
    """temperature_change_1h must be computable from strictly-past rows only --
    it must NOT equal (current_temperature - previous_temperature), since that
    would leak the current-row target value into its own feature."""
    cleaned = clean_weather_data(raw_demo_df)
    feats = build_features(cleaned, lags=[1, 2], windows=[3])
    naive_leaky_version = feats["temperature"] - feats["temperature_lag_1h"]
    # the real (non-leaky) feature should differ from the leaky formula for most rows
    equal_to_leaky = (feats["temperature_change_1h"] - naive_leaky_version).abs() < 1e-9
    assert equal_to_leaky.mean() < 0.5, "temperature_change_1h appears to leak the current row's value"


def test_time_based_split_no_overlap(raw_demo_df):
    cleaned = clean_weather_data(raw_demo_df)
    train, val, test = time_based_split(cleaned, test_days=5, val_days=5)
    if len(train) and len(val):
        assert train["timestamp"].max() <= val["timestamp"].min()
    if len(val) and len(test):
        assert val["timestamp"].max() <= test["timestamp"].min()


def test_geocode_city():
    res = geocode_city("Sydney")
    assert res is not None
    lat, lon, name = res
    assert pytest.approx(lat, 0.1) == -33.8688
    assert name == "Sydney"


def test_baseline_regressor_evaluation():
    y_test = pd.Series([10.0, 12.0, 14.0, 16.0])
    y_lag1 = pd.Series([9.0, 11.0, 13.0, 15.0])
    res = evaluate_persistence_regressor(y_test, y_lag1)
    assert "mae" in res
    assert res["mae"] == 1.0


def test_predictor_location_date():
    predictor = WeatherPredictor()
    res = predictor.predict_location_date("Toronto", target_date="2012-12-31")
    assert res["location"] == "Toronto"
    assert "temperature" in res
    assert "rain_probability" in res
