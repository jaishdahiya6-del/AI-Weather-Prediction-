"""
Appends world-country data to data/raw/weather_data.csv (does not touch/replace
any existing rows -- only adds new ones, on top of the AU/Toronto/live data
already built by build_multi_location_dataset.py).

For each country's capital city, this adds:
  - monthly observations, Jan 2018 -> Dec 2023 (72 months), built from a
    latitude/climate-normal model (seasonal cycle by hemisphere + tropical
    rain-belt adjustment). Tagged is_demo_data=True and
    source_weather_text="Climate-normal model" since these are modeled
    normals, not raw station telemetry -- kept honest per the repo's existing
    is_demo_data convention (see README section 11 / data/README.md).
  - one PRESENT-DAY predicted row (today) per country, computed as this
    month's climate-normal trend plus the small observed warming offset
    already present in the pipeline's real 2026 snapshot cities, i.e. a
    simple, transparent same-pattern-as-history extrapolation.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "raw" / "weather_data.csv"

# country: (capital, lat, lon)
COUNTRIES = {
    "Afghanistan": ("Kabul", 34.5553, 69.2075), "Algeria": ("Algiers", 36.7538, 3.0588),
    "Argentina": ("Buenos Aires", -34.6037, -58.3816), "Armenia": ("Yerevan", 40.1792, 44.4991),
    "Australia": ("Canberra", -35.2809, 149.1300), "Austria": ("Vienna", 48.2082, 16.3738),
    "Bangladesh": ("Dhaka", 23.8103, 90.4125), "Belgium": ("Brussels", 50.8503, 4.3517),
    "Bolivia": ("La Paz", -16.5000, -68.1500), "Brazil": ("Brasilia", -15.7939, -47.8828),
    "Bulgaria": ("Sofia", 42.6977, 23.3219), "Cambodia": ("Phnom Penh", 11.5564, 104.9282),
    "Cameroon": ("Yaounde", 3.8480, 11.5021), "Canada": ("Ottawa", 45.4215, -75.6972),
    "Chile": ("Santiago", -33.4489, -70.6693), "China": ("Beijing", 39.9042, 116.4074),
    "Colombia": ("Bogota", 4.7110, -74.0721), "Costa Rica": ("San Jose", 9.9281, -84.0907),
    "Croatia": ("Zagreb", 45.8150, 15.9819), "Cuba": ("Havana", 23.1136, -82.3666),
    "Czechia": ("Prague", 50.0755, 14.4378), "Denmark": ("Copenhagen", 55.6761, 12.5683),
    "Ecuador": ("Quito", -0.1807, -78.4678), "Egypt": ("Cairo", 30.0444, 31.2357),
    "Ethiopia": ("Addis Ababa", 9.0300, 38.7400), "Finland": ("Helsinki", 60.1699, 24.9384),
    "France": ("Paris", 48.8566, 2.3522), "Germany": ("Berlin", 52.5200, 13.4050),
    "Ghana": ("Accra", 5.6037, -0.1870), "Greece": ("Athens", 37.9838, 23.7275),
    "Guatemala": ("Guatemala City", 14.6349, -90.5069), "Hungary": ("Budapest", 47.4979, 19.0402),
    "Iceland": ("Reykjavik", 64.1466, -21.9426), "India": ("New Delhi", 28.6139, 77.2090),
    "Indonesia": ("Jakarta", -6.2088, 106.8456), "Iran": ("Tehran", 35.6892, 51.3890),
    "Iraq": ("Baghdad", 33.3152, 44.3661), "Ireland": ("Dublin", 53.3498, -6.2603),
    "Israel": ("Jerusalem", 31.7683, 35.2137), "Italy": ("Rome", 41.9028, 12.4964),
    "Jamaica": ("Kingston", 17.9714, -76.7936), "Japan": ("Tokyo", 35.6762, 139.6503),
    "Jordan": ("Amman", 31.9454, 35.9284), "Kazakhstan": ("Astana", 51.1694, 71.4491),
    "Kenya": ("Nairobi", -1.2921, 36.8219), "Kuwait": ("Kuwait City", 29.3759, 47.9774),
    "Laos": ("Vientiane", 17.9757, 102.6331), "Lebanon": ("Beirut", 33.8938, 35.5018),
    "Libya": ("Tripoli", 32.8872, 13.1913), "Malaysia": ("Kuala Lumpur", 3.1390, 101.6869),
    "Mali": ("Bamako", 12.6392, -8.0029), "Mexico": ("Mexico City", 19.4326, -99.1332),
    "Mongolia": ("Ulaanbaatar", 47.8864, 106.9057), "Morocco": ("Rabat", 34.0209, -6.8416),
    "Myanmar": ("Naypyidaw", 19.7633, 96.0785), "Nepal": ("Kathmandu", 27.7172, 85.3240),
    "Netherlands": ("Amsterdam", 52.3676, 4.9041), "New Zealand": ("Wellington", -41.2865, 174.7762),
    "Nicaragua": ("Managua", 12.1150, -86.2362), "Nigeria": ("Abuja", 9.0765, 7.3986),
    "Norway": ("Oslo", 59.9139, 10.7522), "Oman": ("Muscat", 23.5859, 58.4059),
    "Pakistan": ("Islamabad", 33.6844, 73.0479), "Panama": ("Panama City", 8.9824, -79.5199),
    "Paraguay": ("Asuncion", -25.2637, -57.5759), "Peru": ("Lima", -12.0464, -77.0428),
    "Philippines": ("Manila", 14.5995, 120.9842), "Poland": ("Warsaw", 52.2297, 21.0122),
    "Portugal": ("Lisbon", 38.7223, -9.1393), "Qatar": ("Doha", 25.2854, 51.5310),
    "Romania": ("Bucharest", 44.4268, 26.1025), "Russia": ("Moscow", 55.7558, 37.6173),
    "Rwanda": ("Kigali", -1.9403, 30.0619), "Saudi Arabia": ("Riyadh", 24.7136, 46.6753),
    "Senegal": ("Dakar", 14.7167, -17.4677), "Serbia": ("Belgrade", 44.7866, 20.4489),
    "Singapore": ("Singapore", 1.3521, 103.8198), "Slovakia": ("Bratislava", 48.1486, 17.1077),
    "South Africa": ("Pretoria", -25.7479, 28.2293), "South Korea": ("Seoul", 37.5665, 126.9780),
    "Spain": ("Madrid", 40.4168, -3.7038), "Sri Lanka": ("Colombo", 6.9271, 79.8612),
    "Sudan": ("Khartoum", 15.5007, 32.5599), "Sweden": ("Stockholm", 59.3293, 18.0686),
    "Switzerland": ("Bern", 46.9480, 7.4474), "Syria": ("Damascus", 33.5138, 36.2765),
    "Taiwan": ("Taipei", 25.0330, 121.5654), "Tanzania": ("Dodoma", -6.1630, 35.7516),
    "Thailand": ("Bangkok", 13.7563, 100.5018), "Tunisia": ("Tunis", 36.8065, 10.1815),
    "Turkey": ("Ankara", 39.9334, 32.8597), "Uganda": ("Kampala", 0.3476, 32.5825),
    "Ukraine": ("Kyiv", 50.4501, 30.5234), "United Arab Emirates": ("Abu Dhabi", 24.4539, 54.3773),
    "United Kingdom": ("London", 51.5074, -0.1278), "United States": ("Washington DC", 38.9072, -77.0369),
    "Uruguay": ("Montevideo", -34.9011, -56.1645), "Uzbekistan": ("Tashkent", 41.2995, 69.2401),
    "Venezuela": ("Caracas", 10.4806, -66.9036), "Vietnam": ("Hanoi", 21.0278, 105.8342),
    "Yemen": ("Sanaa", 15.3694, 44.1910), "Zambia": ("Lusaka", -15.3875, 28.3228),
    "Zimbabwe": ("Harare", -17.8252, 31.0335),
}


def climate_normal(lat: float, month: int) -> tuple[float, float, float]:
    """Return (temp_c, humidity_pct, rain_prob) for a latitude + calendar month,
    via a seasonal-cycle model: warmer near the equator, seasons flip by hemisphere,
    ITCZ-style rain belt peaks near the equator."""
    # phase: NH warmest ~July(7), SH warmest ~Jan(1)
    phase = (month - 7) if lat >= 0 else (month - 1)
    seasonal = np.cos(2 * np.pi * phase / 12)
    base_temp = 28 - 0.55 * abs(lat)          # equator ~28C, poles much colder
    swing = 3 + 0.35 * abs(lat)               # bigger seasonal swing far from equator
    temp = base_temp + swing * seasonal
    humidity = np.clip(75 - 0.4 * abs(lat) + 10 * np.cos(2 * np.pi * phase / 12), 15, 95)
    rain_prob = np.clip(0.55 - abs(lat) / 120, 0.03, 0.6)
    return round(temp, 1), round(humidity, 1), round(rain_prob, 2)


def build_country_history() -> pd.DataFrame:
    rows = []
    months = pd.date_range("2018-01-01", "2023-12-01", freq="MS")
    for country, (capital, lat, lon) in COUNTRIES.items():
        for m in months:
            temp, hum, rprob = climate_normal(lat, m.month)
            rain = 1 if (hash((country, m.year, m.month)) % 100) / 100 < rprob else 0
            rows.append({
                "timestamp": m, "location": capital, "country": country,
                "latitude": lat, "longitude": lon,
                "temperature": temp, "dew_point": round(temp - (100 - hum) / 5, 1),
                "humidity": hum, "wind_speed": 12.0, "wind_direction": 180,
                "visibility": 15.0, "pressure": 1013.0, "rain": rain,
                "cloud_cover": round(hum * 0.8, 1),
                "rainfall_mm": round(4.0, 1) if rain else 0.0,
                "is_demo_data": True, "source_weather_text": "Climate-normal model",
            })
    return pd.DataFrame(rows)


def build_country_present() -> pd.DataFrame:
    """One 'predicted present' row per country: this month's climate-normal
    trend, nudged by the same small warm offset seen in the real Aug-2026
    live snapshot cities already in the dataset (simple, transparent
    present-day extrapolation, no external API)."""
    today = pd.Timestamp(date.today())
    rows = []
    for country, (capital, lat, lon) in COUNTRIES.items():
        temp, hum, rprob = climate_normal(lat, today.month)
        temp += 0.8  # warming-trend nudge, consistent with 2018-2023 -> 2026 gap
        rain = 1 if rprob > 0.3 else 0
        rows.append({
            "timestamp": today, "location": capital, "country": country,
            "latitude": lat, "longitude": lon,
            "temperature": round(temp, 1), "dew_point": round(temp - (100 - hum) / 5, 1),
            "humidity": hum, "wind_speed": 12.0, "wind_direction": 180,
            "visibility": 15.0, "pressure": 1013.0, "rain": rain,
            "cloud_cover": round(hum * 0.8, 1),
            "rainfall_mm": round(4.0, 1) if rain else 0.0,
            "is_demo_data": True, "source_weather_text": "Predicted (present-day trend)",
        })
    return pd.DataFrame(rows)


def main():
    existing = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    if "country" not in existing.columns:
        existing["country"] = np.nan

    history = build_country_history()
    present = build_country_present()

    combined = pd.concat([existing, history, present], ignore_index=True)
    combined = combined.sort_values(["location", "timestamp"]).reset_index(drop=True)
    combined.to_csv(CSV_PATH, index=False)

    print(f"Added {len(history):,} historical (2018-2023) rows and {len(present):,} present-day predicted rows "
          f"for {len(COUNTRIES)} countries.")
    print(f"Dataset now: {len(combined):,} rows, {combined['location'].nunique()} locations, "
          f"{combined['timestamp'].min()} to {combined['timestamp'].max()}")


if __name__ == "__main__":
    main()
