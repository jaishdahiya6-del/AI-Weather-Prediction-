"""
Transform the real 'Rain in Australia' (BOM) dataset into this project's
expected schema and save it to data/raw/weather_data.csv.

Source: Australian Bureau of Meteorology daily observations, 2007-2017,
via the public 'Rain in Australia' dataset (weatherAUS.csv), ~145k daily
rows across 46 real Australian weather stations. This is REAL historical
weather data, not synthetic.

Notes on the transform (documented, not hidden):
- Data is DAILY (one row per station per day), so lag/rolling features
  labelled "_1h" etc. in this project actually mean "1 day", "3 days" etc.
  when used on this dataset -- the code doesn't care about the unit, only
  that it's the correct number of PAST rows, so nothing leaks. Just don't
  read "1h" literally when using this dataset.
- temperature = mean(Temp9am, Temp3pm), pressure = mean(Pressure9am, Pressure3pm),
  humidity = mean(Humidity9am, Humidity3pm), wind_speed = mean(WindSpeed9am, WindSpeed3pm)
- cloud_cover: source is in oktas (0-8) -> scaled to a 0-100% approximation
- visibility & dew_point are NOT in the source data and are estimated
  (dew_point via the Magnus formula from temperature+humidity; visibility
  approximated from cloud cover). These two columns are lower-confidence
  than the directly-observed ones.
- rain = RainToday (Yes/No -> 1/0), rainfall_mm = Rainfall (mm) -- both are
  genuine BOM observations.
- latitude/longitude are approximate town-centre coordinates for each station.
"""
import numpy as np
import pandas as pd

LOCATION_COORDS = {
    "Adelaide": (-34.9285, 138.6007), "Albany": (-35.0275, 117.8840), "Albury": (-36.0737, 146.9135),
    "AliceSprings": (-23.6980, 133.8807), "BadgerysCreek": (-33.8828, 150.7273), "Ballarat": (-37.5622, 143.8503),
    "Bendigo": (-36.7570, 144.2794), "Brisbane": (-27.4698, 153.0251), "Cairns": (-16.9186, 145.7781),
    "Canberra": (-35.2809, 149.1300), "Cobar": (-31.4958, 145.8389), "CoffsHarbour": (-30.2963, 153.1157),
    "Dartmoor": (-37.9200, 141.2700), "Darwin": (-12.4634, 130.8456), "GoldCoast": (-28.0167, 153.4000),
    "Hobart": (-42.8821, 147.3272), "Launceston": (-41.4332, 147.1441), "Melbourne": (-37.8136, 144.9631),
    "MelbourneAirport": (-37.6690, 144.8410), "Mildura": (-34.2080, 142.1246), "Moree": (-29.4658, 149.8339),
    "MountGambier": (-37.8284, 140.7828), "MountGinini": (-35.5297, 148.7723), "Newcastle": (-32.9283, 151.7817),
    "NorahHead": (-33.2833, 151.5667), "NorfolkIsland": (-29.0408, 167.9547), "Nuriootpa": (-34.4700, 138.9950),
    "PearceRAAF": (-31.6675, 116.0169), "Penrith": (-33.7506, 150.6944), "Perth": (-31.9505, 115.8605),
    "PerthAirport": (-31.9385, 115.9672), "Portland": (-38.3450, 141.6041), "Richmond": (-33.6000, 150.7500),
    "Sale": (-38.1050, 147.0670), "SalmonGums": (-32.9810, 121.6440), "Sydney": (-33.8688, 151.2093),
    "SydneyAirport": (-33.9399, 151.1753), "Townsville": (-19.2590, 146.8169), "Tuggeranong": (-35.4244, 149.0888),
    "WaggaWagga": (-35.1150, 147.3670), "Walpole": (-34.9780, 116.7310), "Watsonia": (-37.7110, 145.0830),
    "Williamtown": (-32.8150, 151.8430), "Witchcliffe": (-34.0260, 115.1000), "Wollongong": (-34.4278, 150.8931),
    "Woomera": (-31.1990, 136.8290),
}


def dew_point_magnus(temp_c, rh_pct):
    a, b = 17.62, 243.12
    rh = np.clip(rh_pct, 1, 100) / 100.0
    gamma = np.log(rh) + (a * temp_c) / (b + temp_c)
    return (b * gamma) / (a - gamma)


def transform(src_path: str, dst_path: str):
    df = pd.read_csv(src_path)

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df["Date"])
    out["location"] = df["Location"]
    out["latitude"] = df["Location"].map(lambda l: LOCATION_COORDS.get(l, (np.nan, np.nan))[0])
    out["longitude"] = df["Location"].map(lambda l: LOCATION_COORDS.get(l, (np.nan, np.nan))[1])

    out["temperature"] = df[["Temp9am", "Temp3pm"]].mean(axis=1)
    out["temperature"] = out["temperature"].fillna(df[["MinTemp", "MaxTemp"]].mean(axis=1))
    out["humidity"] = df[["Humidity9am", "Humidity3pm"]].mean(axis=1)
    out["pressure"] = df[["Pressure9am", "Pressure3pm"]].mean(axis=1)
    out["wind_speed"] = df[["WindSpeed9am", "WindSpeed3pm"]].mean(axis=1)
    out["wind_speed"] = out["wind_speed"].fillna(df["WindGustSpeed"])
    out["wind_direction"] = df["WindDir3pm"].fillna(df["WindDir9am"])

    compass = {d: i * 22.5 for i, d in enumerate(
        ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    )}
    out["wind_direction"] = out["wind_direction"].map(compass)

    cloud_okta = df[["Cloud9am", "Cloud3pm"]].mean(axis=1)
    out["cloud_cover"] = (cloud_okta / 8.0 * 100).clip(0, 100)

    # Not present in source -- estimated (see module docstring)
    out["visibility"] = (20 - out["cloud_cover"] / 10).clip(0.5, 20)
    out["dew_point"] = dew_point_magnus(out["temperature"], out["humidity"])

    out["rain"] = df["RainToday"].map({"Yes": 1, "No": 0})
    out["rainfall_mm"] = df["Rainfall"]

    out = out.dropna(subset=["timestamp", "location", "temperature", "humidity", "rain"])
    out = out.sort_values(["location", "timestamp"]).reset_index(drop=True)
    out.to_csv(dst_path, index=False)
    print(f"Wrote {len(out):,} real rows -> {dst_path}")
    print(f"Locations: {out['location'].nunique()}, date range: {out['timestamp'].min()} to {out['timestamp'].max()}")


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "weatherAUS.csv"
    dst = sys.argv[2] if len(sys.argv) > 2 else "data/raw/weather_data.csv"
    transform(src, dst)
