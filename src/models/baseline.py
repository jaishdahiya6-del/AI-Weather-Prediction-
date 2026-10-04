"""
Baseline models for weather forecasting:
1. Persistence Model (Naive Forecast): Predicts current observation or t-1 value for next time step.
2. Linear / Logistic Regression Baseline: Standard linear models using basic numeric features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.evaluation.metrics import classification_report_dict, regression_report_dict
from src.utils.logger import get_logger

log = get_logger(__name__)


def evaluate_persistence_classifier(y_test: pd.Series, y_lag1: pd.Series) -> dict:
    """Persistence baseline for rain classification: predicts rain = lag_1h rain."""
    preds = y_lag1.fillna(0).astype(int)
    return classification_report_dict(y_test, preds)


def evaluate_persistence_regressor(y_test: pd.Series, y_lag1: pd.Series) -> dict:
    """Persistence baseline for continuous targets: predicts value at time t = lag_1h value."""
    preds = y_lag1.ffill().bfill().values
    return regression_report_dict(y_test, preds)


def evaluate_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """Linear Regression baseline model."""
    model = LinearRegression()
    model.fit(X_train.fillna(0), y_train)
    preds = model.predict(X_test.fillna(0))
    return regression_report_dict(y_test, preds)


def evaluate_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """Logistic Regression baseline classifier."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train.fillna(0), y_train)
    preds = model.predict(X_test.fillna(0))
    probs = model.predict_proba(X_test.fillna(0))[:, 1] if hasattr(model, "predict_proba") else None
    return classification_report_dict(y_test, preds, y_proba=probs)
