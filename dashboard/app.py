"""
Streamlit dashboard for the AI Weather Prediction & Forecasting System.

Run with:
    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.cleaner import clean_weather_data
from src.data.loader import load_raw_data
from src.prediction.predictor import WeatherPredictor
from src.utils.config import load_config, resolve_path

st.set_page_config(page_title="AI Weather Forecasting System", page_icon="🌦️", layout="wide")

CFG = load_config()


@st.cache_data(show_spinner="Loading and cleaning weather data...")
def get_clean_data():
    raw = load_raw_data(CFG)
    return clean_weather_data(raw)


def demo_banner(df: pd.DataFrame):
    if df["is_demo_data"].any():
        st.warning(
            "⚠️ **DEMO DATA MODE** — no real dataset found at `data/raw/weather_data.csv`. "
            "All charts and predictions below are generated from synthetic statistical patterns, "
            "NOT real observations. Place a real dataset to get real results (see `data/README.md`).",
            icon="⚠️",
        )


def load_metrics(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------- Sidebar
st.sidebar.title("🌦️ Weather AI")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🌦️ Prediction", "📅 Forecast", "📊 Analytics", "🤖 Model Performance", "🌍 Map"],
)

df = get_clean_data()
locations = sorted(df["location"].unique().tolist())
selected_location = st.sidebar.selectbox("Location", locations)

# ---------------------------------------------------------------- Home
if page == "🏠 Home":
    st.title("AI Weather Prediction & Forecasting System")
    st.caption("Research / educational forecasting system — not a replacement for official meteorological warnings.")
    demo_banner(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Observations", f"{len(df):,}")
    c2.metric("Locations", df["location"].nunique())
    c3.metric("Date Range (days)", (df["timestamp"].max() - df["timestamp"].min()).days)
    c4.metric("Missing Values", int(df.isna().sum().sum()))

    st.subheader(f"Latest observation — {selected_location}")
    latest = df[df["location"] == selected_location].sort_values("timestamp").iloc[-1]
    cols = st.columns(5)
    cols[0].metric("🌡️ Temperature", f"{latest['temperature']:.1f} °C")
    cols[1].metric("💧 Humidity", f"{latest['humidity']:.0f} %")
    cols[2].metric("💨 Wind Speed", f"{latest['wind_speed']:.1f} km/h")
    cols[3].metric("🌫️ Pressure", f"{latest['pressure']:.0f} hPa")
    cols[4].metric("☁️ Cloud Cover", f"{latest['cloud_cover']:.0f} %")

    clf_metrics = load_metrics(resolve_path("models/classification/metrics.json"))
    st.info(
        "✅ Trained models found — go to **Prediction** for a live forecast."
        if clf_metrics else
        "ℹ️ No trained models yet. Run `python run_pipeline.py` first, then reload this dashboard."
    )

# ---------------------------------------------------------------- Prediction
elif page == "🌦️ Prediction":
    st.title("🌦️ Weather Prediction")
    demo_banner(df)
    predictor = WeatherPredictor(CFG)

    col_loc, col_date = st.columns([2, 1])
    with col_loc:
        custom_city = st.text_input("Enter City/Location Name", value=selected_location)
    with col_date:
        use_custom_date = st.checkbox("Select Target Date", value=False)
        target_date_val = st.date_input("Target Date", value=datetime.now().date()) if use_custom_date else None

    if st.button("Generate Forecast", type="primary"):
        try:
            target_date_str = target_date_val.strftime("%Y-%m-%d") if target_date_val else None
            result = predictor.predict_location_date(custom_city, target_date_str)
            if result.get("is_demo_data"):
                st.warning("Prediction generated from DEMO DATA — not a real forecast.")
            st.caption(f"Forecast for **{result['location']}** based on historical data as of {result['as_of']}")

            c1, c2, c3 = st.columns(3)
            c1.metric("🌧️ Rain Probability", f"{result.get('rain_probability', 0):.0f}%")
            c2.metric("🌡️ Temperature", f"{result.get('temperature', float('nan')):.1f} °C")
            c3.metric("💨 Wind Speed", f"{result.get('wind_speed', float('nan')):.1f} km/h")

            c4, c5 = st.columns(2)
            c4.metric("💧 Humidity", f"{result.get('humidity', float('nan')):.1f} %")
            c5.metric("🌧️ Rainfall", f"{result.get('rainfall_mm', float('nan')):.1f} mm")

            verdict = "🌧️ HIGH CHANCE OF RAIN" if result.get("rain_probability", 0) >= 50 else "☀️ LOW CHANCE OF RAIN"
            st.subheader(verdict)
        except FileNotFoundError as e:
            st.error(str(e))
        except ValueError as e:
            st.error(str(e))

# ---------------------------------------------------------------- Forecast
elif page == "📅 Forecast":
    st.title("📅 Recent Trend (proxy for forecast view)")
    demo_banner(df)
    loc_df = df[df["location"] == selected_location].sort_values("timestamp").tail(24 * 7)
    st.caption("Showing the most recent 7 days of historical data for this location.")

    for col, label in [("temperature", "Temperature (°C)"), ("rain", "Rain (0/1)"),
                        ("rainfall_mm", "Rainfall (mm)"), ("wind_speed", "Wind Speed (km/h)"),
                        ("humidity", "Humidity (%)"), ("pressure", "Pressure (hPa)")]:
        fig = px.line(loc_df, x="timestamp", y=col, title=label)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- Analytics
elif page == "📊 Analytics":
    st.title("📊 Historical Analytics")
    demo_banner(df)

    tab1, tab2, tab3 = st.tabs(["Distributions", "Seasonal", "Correlation"])
    with tab1:
        for col in ["temperature", "humidity", "wind_speed", "pressure", "rainfall_mm"]:
            st.plotly_chart(px.histogram(df, x=col, color="location", title=f"{col} distribution"),
                             use_container_width=True)
    with tab2:
        monthly = df.groupby([df["timestamp"].dt.month, "location"])["rainfall_mm"].mean().reset_index()
        monthly.columns = ["month", "location", "avg_rainfall_mm"]
        st.plotly_chart(px.line(monthly, x="month", y="avg_rainfall_mm", color="location",
                                 title="Average Monthly Rainfall"), use_container_width=True)
        hourly = df.groupby([df["timestamp"].dt.hour, "location"])["temperature"].mean().reset_index()
        hourly.columns = ["hour", "location", "avg_temperature"]
        st.plotly_chart(px.line(hourly, x="hour", y="avg_temperature", color="location",
                                 title="Average Hourly Temperature"), use_container_width=True)
    with tab3:
        numeric_cols = ["temperature", "humidity", "pressure", "wind_speed", "cloud_cover", "visibility", "rainfall_mm"]
        corr = df[numeric_cols].corr()
        st.plotly_chart(px.imshow(corr, text_auto=".2f", title="Correlation Heatmap"), use_container_width=True)

    st.subheader("🌀 Radial temperature spread — top 20 hottest locations")
    top20 = (df.groupby("location")["temperature"].mean().sort_values(ascending=False).head(20).reset_index())
    fig_polar = go.Figure(go.Barpolar(
        r=top20["temperature"], theta=top20["location"],
        marker=dict(color=top20["temperature"], colorscale="Turbo", line=dict(color="white", width=1)),
    ))
    fig_polar.update_layout(
        template="plotly_dark", showlegend=False, height=560,
        polar=dict(radialaxis=dict(showticklabels=True, ticksuffix="°C")),
        title="Average temperature by location (polar)",
    )
    st.plotly_chart(fig_polar, use_container_width=True)

# ---------------------------------------------------------------- Model Performance
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")
    clf_metrics = load_metrics(resolve_path("models/classification/metrics.json"))
    reg_metrics = load_metrics(resolve_path("models/regression/metrics.json"))
    lstm_metrics = load_metrics(resolve_path("models/deep_learning/metrics.json"))

    if not any([clf_metrics, reg_metrics, lstm_metrics]):
        st.info("No trained models found yet. Run `python run_pipeline.py`.")
    if clf_metrics:
        st.subheader("Rain Classification")
        st.dataframe(pd.DataFrame(clf_metrics["results"]).T)
        st.caption(f"Best model: **{clf_metrics['best_model']}**")
    if reg_metrics:
        st.subheader("Regression Models")
        for target, data in reg_metrics.items():
            st.markdown(f"**{target}** — best: `{data['best_model']}`")
            st.dataframe(pd.DataFrame(data["results"]).T)
    if lstm_metrics:
        st.subheader("LSTM")
        st.json(lstm_metrics)

# ---------------------------------------------------------------- Map
elif page == "🌍 Map":
    st.title("🌍 Weather Observation Map")
    st.caption(f"{df['location'].nunique()} real locations · data spans "
               f"{df['timestamp'].min():%Y-%m-%d} to {df['timestamp'].max():%Y-%m-%d}, "
               "including a live snapshot captured 2026-08-09.")
    demo_banner(df)
    latest_per_loc = df.sort_values("timestamp").groupby("location").tail(1).copy()
    latest_per_loc["size"] = latest_per_loc["temperature"] - latest_per_loc["temperature"].min() + 1

    proj = st.radio("Projection", ["orthographic (globe)", "natural earth (flat)"], horizontal=True)
    fig = px.scatter_geo(
        latest_per_loc, lat="latitude", lon="longitude", hover_name="location",
        size="size", color="temperature", color_continuous_scale="Turbo",
        projection="orthographic" if proj.startswith("orthographic") else "natural earth",
        title="Latest observation per location — bubble size/color = temperature (°C)",
    )
    fig.update_geos(showocean=True, oceancolor="#0b1220", landcolor="#16213a",
                     showcountries=True, countrycolor="#2a3a5c", bgcolor="rgba(0,0,0,0)")
    fig.update_layout(height=600, paper_bgcolor="rgba(0,0,0,0)", font_color="#dce8f5")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔥 Global temperature race — by year")
    yearly = df.copy()
    yearly["year"] = yearly["timestamp"].dt.year
    yearly = yearly.groupby(["year", "location"], as_index=False).agg(
        temperature=("temperature", "mean"), latitude=("latitude", "first"), longitude=("longitude", "first"))
    yearly["size"] = yearly["temperature"] - yearly["temperature"].min() + 1
    fig2 = px.scatter_geo(
        yearly.sort_values("year"), lat="latitude", lon="longitude", hover_name="location",
        size="size", color="temperature", color_continuous_scale="Turbo",
        animation_frame="year", projection="natural earth",
        title="Average temperature by location, animated by year",
    )
    fig2.update_layout(height=560)
    st.plotly_chart(fig2, use_container_width=True)
