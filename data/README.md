# Dataset Setup

**A real dataset is already here**: `data/raw/weather_data.csv` — 8,784
hourly observations for Toronto, 2012 (Environment Canada, via a public
GitHub mirror). Every column's provenance (real vs derived/proxy) is
documented at the top of `scripts/prepare_real_dataset.py` — read it before
trusting any specific column, especially `cloud_cover` (proxy) and
`rainfall_mm` (not available in source, always 0.0).

To swap in your own real dataset, overwrite that file with the same
columns, or adapt `scripts/prepare_real_dataset.py` for your source format.

Required columns (see `config/config.yaml` to rename/remap):

`timestamp, location, latitude, longitude, temperature, humidity, pressure,
wind_speed, wind_direction, cloud_cover, visibility, dew_point, rain, rainfall_mm`

Good sources: NOAA Integrated Surface Database, Open-Meteo Historical API,
Kaggle "Weather Prediction Dataset" / "Rain in Australia", ERA5 reanalysis.

**If no file is present, the pipeline automatically falls back to a
synthetic DEMO dataset** (see `src/data/loader.py`) so you can run and test
the whole system immediately. Every demo row is tagged `is_demo_data=True`
and the dashboard displays a visible "DEMO DATA" banner whenever it is used.
Do not treat demo-mode predictions as real forecasts.
