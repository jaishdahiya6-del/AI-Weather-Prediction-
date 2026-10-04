"""
Open-Meteo API Fetcher for historical and forecast weather data.

Documentation: Open-Meteo is a free, non-commercial open-source weather API that requires
no API key for basic usage. https://open-meteo.com/
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests

from src.utils.logger import get_logger

log = get_logger(__name__)

# Known major cities and their coordinates
CITY_COORDINATES = {
    "sydney": (-33.8688, 151.2093),
    "melbourne": (-37.8136, 144.9631),
    "brisbane": (-27.4698, 153.0251),
    "perth": (-31.9505, 115.8605),
    "adelaide": (-34.9285, 138.6007),
    "canberra": (-35.2809, 149.1300),
    "hobart": (-42.8821, 147.3272),
    "darwin": (-12.4634, 130.8456),
    "toronto": (43.6532, -79.3832),
    "london": (51.5074, -0.1278),
    "new york": (40.7128, -74.0060),
    "tokyo": (35.6762, 139.6503),
    "paris": (48.8566, 2.3522),
    "berlin": (52.5200, 13.4050),
}


def geocode_city(city_name: str) -> Optional[Tuple[float, float, str]]:
    """Geocode city name using Open-Meteo Geocoding API or fallback dictionary."""
    city_lower = city_name.strip().lower()
    if city_lower in CITY_COORDINATES:
        lat, lon = CITY_COORDINATES[city_lower]
        return lat, lon, city_name.title()

    try:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if "results" in data and len(data["results"]) > 0:
                result = data["results"][0]
                return float(result["latitude"]), float(result["longitude"]), result["name"]
    except Exception as e:
        log.warning(f"Geocoding failed for {city_name}: {e}")

    return None


def fetch_open_meteo_historical(
    city_name: str,
    start_date: str,
    end_date: str,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch historical weather data from Open-Meteo API.

    Data Source Documented: Open-Meteo Historical Weather API (ERA5 reanalysis / Seamless hourly)
    URL: https://archive-api.open-meteo.com/v1/archive
    """
    coords = geocode_city(city_name)
    if not coords:
        raise ValueError(f"Could not resolve coordinates for city: {city_name}")

    lat, lon, resolved_name = coords
    log.info(f"Fetching Open-Meteo data for {resolved_name} ({lat}, {lon}) from {start_date} to {end_date}")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
            "cloud_cover",
            "visibility",
            "dew_point_2m",
            "precipitation",
            "rain"
        ],
        "timezone": "UTC"
    }
    if api_key:
        params["apikey"] = api_key

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if "hourly" not in data:
            raise ValueError(f"No hourly data returned from Open-Meteo for {city_name}")

        hourly = data["hourly"]
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(hourly["time"]),
            "location": resolved_name,
            "latitude": lat,
            "longitude": lon,
            "temperature": hourly.get("temperature_2m"),
            "humidity": hourly.get("relative_humidity_2m"),
            "pressure": hourly.get("surface_pressure"),
            "wind_speed": hourly.get("wind_speed_10m"),
            "cloud_cover": hourly.get("cloud_cover"),
            "visibility": [v / 1000.0 if v is not None else None for v in hourly.get("visibility", [])],  # m to km
            "dew_point": hourly.get("dew_point_2m"),
            "rainfall_mm": hourly.get("precipitation", [0.0] * len(hourly["time"])),
            "rain": [(1 if (p is not None and p > 0.1) else 0) for p in hourly.get("precipitation", [])],
            "is_demo_data": False,
        })
        return df

    except Exception as e:
        log.error(f"Error fetching data from Open-Meteo: {e}")
        raise
