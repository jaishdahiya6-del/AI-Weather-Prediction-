"""Time-based splitting utilities. Never shuffle time-series data before splitting."""
from __future__ import annotations

import pandas as pd


def time_based_split(df: pd.DataFrame, test_days: int, val_days: int, ts_col: str = "timestamp"):
    """Split chronologically: [ ... train ... | val | test ] by wall-clock time.
    This guarantees the model is only ever evaluated on data strictly after
    what it trained on -- the correct validation strategy for forecasting."""
    df = df.sort_values(ts_col)
    max_ts = df[ts_col].max()
    test_start = max_ts - pd.Timedelta(days=test_days)
    val_start = test_start - pd.Timedelta(days=val_days)

    train = df[df[ts_col] < val_start]
    val = df[(df[ts_col] >= val_start) & (df[ts_col] < test_start)]
    test = df[df[ts_col] >= test_start]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
