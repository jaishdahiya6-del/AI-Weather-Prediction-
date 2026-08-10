"""
Prepares the bundled REAL weather dataset (data/raw/weather_data.csv) from its
original source so the project has real predictions out of the box.

SOURCE: Environment Canada hourly observations for 2012, Toronto (Pearson
Airport station), redistributed as a CSV at:
https://github.com/raunak274/analyzing-weather-dataset (file.csv)
This is the same well-known "weather_2012.csv" dataset used in many
pandas/numpy tutorials (8,784 hourly rows, Jan-Dec 2012, single station).

WHAT IS REAL vs DERIVED in the output file:
- timestamp, temperature, dew_point, humidity, wind_speed, visibility,
  pressure: REAL observations from the source, unit-converted where needed
  (pressure kPa -> hPa).
- location/latitude/longitude: set to "Toronto" / Pearson Airport
  coordinates, since the source file is single-station and doesn't include
  location metadata itself.
- rain (0/1): DERIVED from the source's free-text `Weather` column (1 if it
  contains "Rain" or "Drizzle", else 0). This is a real historical fact
  about that hour, just re-encoded from text to a boolean.
- cloud_cover (%): DERIVED/APPROXIMATED from the same `Weather` text
  description (e.g. "Clear"->0, "Mainly Clear"->20, "Mostly Cloudy"->70,
  "Cloudy"->95) since the source has no numeric cloud-cover field. This is
  a coarse proxy, not a real sensor reading -- treat cloud_cover values
  from this dataset as approximate.
- rainfall_mm: The source has no precipitation-amount field at all. Rather
  than inventing numbers, this is left as 0.0 for rain=0 hours; for rain=1
  hours it is NOT a real measurement -- see the loud warning printed by
  this script and the note in data/README.md. If you need real rainfall
  amounts, source a dataset that includes precipitation (e.g. an ERA5 or
  NOAA extract) and swap it in.
- wind_direction: not present in the source; filled with NaN then dropped
  by the cleaning pipeline's per-location NaN handling (not used by any
  current model).

Usage:
    python scripts/prepare_real_dataset.py <path_to_downloaded_file.csv> <output_path>
"""
from __future__ import annotations

import sys

import pandas as pd

CLOUD_COVER_MAP = {
    "clear": 0, "mainly clear": 20, "mostly cloudy": 70, "cloudy": 95,
}


def cloud_cover_from_text(weather_text: str) -> float:
    text = str(weather_text).lower()
    for key, val in CLOUD_COVER_MAP.items():
        if key in text:
            return val
    # anything else (fog/rain/snow/etc without an explicit sky descriptor)
    return 60.0


def main(src_path: str, out_path: str):
    df = pd.read_csv(src_path)
    df.columns = [c.strip() for c in df.columns]

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df["Date/Time"])
    out["location"] = "Toronto"
    out["latitude"] = 43.6777
    out["longitude"] = -79.6248
    out["temperature"] = df["Temp (C)"]
    out["dew_point"] = df["Dew Point Temp (C)"]
    out["humidity"] = df["Rel Hum (%)"]
    out["wind_speed"] = df["Wind Spd (km/h)"]
    out["wind_direction"] = pd.NA
    out["visibility"] = df["Visibility (km)"]
    out["pressure"] = df["Stn Press (kPa)"] * 10  # kPa -> hPa
    out["rain"] = df["Weather"].str.contains("Rain|Drizzle", case=False, na=False).astype(int)
    out["cloud_cover"] = df["Weather"].apply(cloud_cover_from_text)
    out["rainfall_mm"] = 0.0  # see module docstring: source has no real precip amounts
    out["is_demo_data"] = False
    out["source_weather_text"] = df["Weather"]  # kept for transparency/auditing

    out = out.sort_values("timestamp").reset_index(drop=True)
    out.to_csv(out_path, index=False)

    print(f"Wrote {len(out):,} rows to {out_path}")
    print("NOTE: rainfall_mm is NOT a real measurement in this source dataset -- "
          "it is 0.0 everywhere. Rain occurrence (the `rain` column) IS real. "
          "See this script's docstring for full provenance of every column.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/prepare_real_dataset.py <src_csv> <out_csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
