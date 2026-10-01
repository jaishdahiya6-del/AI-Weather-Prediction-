"""
Data Exploration and Visualization script.

Generates and saves exploratory data analysis charts (distributions, temporal trends,
correlation heatmaps) to reports/figures/.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.cleaner import clean_weather_data
from src.data.loader import load_raw_data
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

log = get_logger(__name__)


def explore_and_visualize():
    cfg = load_config()
    raw = load_raw_data(cfg)
    df = clean_weather_data(raw, save_processed=True, cfg=cfg)

    figures_dir = resolve_path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # 1. Weather Variables Distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    cols = ["temperature", "humidity", "pressure", "wind_speed", "cloud_cover", "rainfall_mm"]
    titles = ["Temperature (°C)", "Humidity (%)", "Pressure (hPa)", "Wind Speed (km/h)", "Cloud Cover (%)", "Rainfall (mm)"]

    for i, col in enumerate(cols):
        if col in df.columns:
            sns.histplot(df[col], kde=True, ax=axes[i], color="teal", bins=30)
            axes[i].set_title(f"Distribution of {titles[i]}")
            axes[i].set_xlabel(titles[i])
            axes[i].set_ylabel("Count")

    plt.tight_layout()
    dist_path = figures_dir / "eda_variable_distributions.png"
    plt.savefig(dist_path, dpi=300)
    plt.close()
    log.info(f"Saved distribution plot to {dist_path}")

    # 2. Monthly Weather Trends
    df["month"] = df["timestamp"].dt.month
    monthly_stats = df.groupby("month")[["temperature", "humidity", "rainfall_mm"]].mean().reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    sns.lineplot(data=monthly_stats, x="month", y="temperature", ax=ax1, color="firebrick", marker="o", label="Avg Temp (°C)")
    sns.lineplot(data=monthly_stats, x="month", y="humidity", ax=ax1, color="navy", marker="s", label="Avg Humidity (%)")
    sns.barplot(data=monthly_stats, x="month", y="rainfall_mm", ax=ax2, alpha=0.3, color="dodgerblue", label="Avg Rainfall (mm)")

    ax1.set_xlabel("Month")
    ax1.set_ylabel("Temperature (°C) / Humidity (%)")
    ax2.set_ylabel("Rainfall (mm)")
    ax1.set_title("Seasonal Monthly Weather Trends")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    trend_path = figures_dir / "eda_monthly_trends.png"
    plt.savefig(trend_path, dpi=300)
    plt.close()
    log.info(f"Saved monthly trends plot to {trend_path}")

    # 3. Correlation Heatmap
    plt.figure(figsize=(9, 7))
    num_cols = [c for c in cols if c in df.columns]
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Weather Feature Correlation Heatmap")
    plt.tight_layout()
    corr_path = figures_dir / "eda_correlation_heatmap.png"
    plt.savefig(corr_path, dpi=300)
    plt.close()
    log.info(f"Saved correlation heatmap to {corr_path}")

    # Print summary statistics
    print("\n=== Weather Dataset Summary Statistics ===")
    print(df[num_cols].describe().T[["mean", "std", "min", "50%", "max"]])


if __name__ == "__main__":
    explore_and_visualize()
