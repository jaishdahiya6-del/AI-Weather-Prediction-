"""
Train an LSTM to forecast multiple weather variables one step ahead from a
sequence of past hourly observations.

Sequences are built per-location so no data crosses location boundaries, and
train/val/test are split chronologically (never shuffled) before sequencing.

Usage:
    python -m src.training.train_lstm
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Model

from src.data.cleaner import clean_weather_data
from src.data.loader import load_raw_data
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger
from src.utils.splitting import time_based_split

log = get_logger(__name__)

SEQ_FEATURES = ["temperature", "humidity", "pressure", "wind_speed", "cloud_cover", "rainfall_mm"]
TARGETS = ["temperature", "humidity", "wind_speed", "rainfall_mm"]


def make_sequences(df, seq_len, feature_cols, target_cols):
    """Build (X, y) sequences per-location. X[i] = seq_len past hourly rows,
    y[i] = target values at the next hour. Purely backward-looking -> no leakage."""
    X, y = [], []
    for _, group in df.groupby("location"):
        group = group.sort_values("timestamp")
        feats = group[feature_cols].values
        targs = group[target_cols].values
        for i in range(len(group) - seq_len):
            X.append(feats[i:i + seq_len])
            y.append(targs[i + seq_len])
    if not X:
        return np.empty((0, seq_len, len(feature_cols))), np.empty((0, len(target_cols)))
    return np.array(X), np.array(y)


def build_model(seq_len, n_features, n_targets, units):
    inputs = Input(shape=(seq_len, n_features))
    x = LSTM(units, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(units // 2)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(n_targets)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train():
    cfg = load_config()
    tf.random.set_seed(cfg["random_seed"])

    raw = load_raw_data(cfg)
    clean = clean_weather_data(raw)

    train_df, val_df, test_df = time_based_split(
        clean, cfg["training"]["test_size_days"], cfg["training"]["val_size_days"]
    )

    scaler = StandardScaler()
    scaler.fit(train_df[SEQ_FEATURES])

    train_df = train_df.copy()
    train_df[SEQ_FEATURES] = scaler.transform(train_df[SEQ_FEATURES])

    seq_len = cfg["lstm"]["sequence_length"]
    X_train, y_train = make_sequences(train_df, seq_len, SEQ_FEATURES, TARGETS)

    has_val = len(val_df) > 0
    if has_val:
        val_df = val_df.copy()
        val_df[SEQ_FEATURES] = scaler.transform(val_df[SEQ_FEATURES])
        X_val, y_val = make_sequences(val_df, seq_len, SEQ_FEATURES, TARGETS)
    else:
        split_idx = int(len(X_train) * 0.85)
        X_val, y_val = X_train[split_idx:], y_train[split_idx:]
        X_train, y_train = X_train[:split_idx], y_train[:split_idx]

    has_test = len(test_df) > 0
    if has_test:
        test_df = test_df.copy()
        test_df[SEQ_FEATURES] = scaler.transform(test_df[SEQ_FEATURES])
        X_test, y_test = make_sequences(test_df, seq_len, SEQ_FEATURES, TARGETS)
    else:
        X_test, y_test = X_val, y_val

    if len(X_test) == 0:
        X_test, y_test = X_val, y_val

    log.info(f"Sequences -> train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")

    out_dir = resolve_path("models/deep_learning")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(seq_len, len(SEQ_FEATURES), len(TARGETS), cfg["lstm"]["units"])
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ModelCheckpoint(str(out_dir / "lstm_weather.keras"), save_best_only=True, monitor="val_loss"),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg["lstm"]["epochs"],
        batch_size=cfg["lstm"]["batch_size"],
        callbacks=callbacks,
        verbose=2,
    )

    eval_results = model.evaluate(X_test, y_test, verbose=0)
    test_loss = float(eval_results[0])
    test_mae = float(eval_results[1])
    log.info(f"Test MSE={test_loss:.4f} Test MAE={test_mae:.4f}")

    joblib.dump(scaler, out_dir / "scaler.joblib")
    joblib.dump({"features": SEQ_FEATURES, "targets": TARGETS, "seq_len": seq_len}, out_dir / "lstm_metadata.joblib")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"test_mse": test_loss, "test_mae": test_mae}, f, indent=2)

    return {"test_mse": test_loss, "test_mae": test_mae}


if __name__ == "__main__":
    train()
