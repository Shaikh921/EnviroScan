"""
EnviroScan Dashboard
Real-Time Air Pollution Monitoring System
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import io
import logging
from datetime import datetime
import folium
from streamlit_folium import st_folium

# Import project configuration
from config import (
    OPENWEATHER_API_KEY, FINAL_DATASET_BALANCED, XGBOOST_MODEL, LABEL_ENCODER,
    POLLUTION_MAP_HTML, CONFUSION_MATRIX_XGBOOST, CONFUSION_MATRIX_RF,
    ALL_FEATURES, get_aqi_status
)
from utils import fetch_weather_data, fetch_pollution_data, logger

# Configure logging
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="EnviroScan Dashboard",
    layout="wide",
    page_icon="🌍"
)

st.title("🌍 EnviroScan: Real-Time Air Pollution Monitoring System")
st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ------------------------------------------------
# VALIDATE API KEY
# ------------------------------------------------

if not OPENWEATHER_API_KEY:
    st.error("⚠️ OpenWeather API key not found. Please set OPENWEATHER_KEY in .env file")
    st.stop()

# ------------------------------------------------
# LOAD DATASET
# ------------------------------------------------

@st.cache_data
def load_data():
    """Load and cache the pollution dataset"""
    try:
        df = pd.read_csv(FINAL_DATASET_BALANCED)
        df.columns = df.columns.str.strip()
        logger.info(f"Dataset loaded: {df.shape}")
        return df
    except FileNotFoundError:
        st.error(f"Dataset not found: {FINAL_DATASET_BALANCED}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

df = load_data()

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

@st.cache_resource
def load_model():
    """Load and cache the trained model and encoder"""
    try:
        model = joblib.load(XGBOOST_MODEL)
        encoder = joblib.load(LABEL_ENCODER)
        logger.info("Model and encoder loaded successfully")
        return model, encoder
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model, encoder = load_model()

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.header("📊 Dashboard Controls")

city = st.sidebar.selectbox(
    "Select City",
    sorted(df["city"].unique())
)

city_row = df[df["city"] == city].iloc[0]

lat = city_row["latitude"]
lon = city_row["longitude"]

# ------------------------------------------------
# FETCH REAL-TIME DATA
# ------------------------------------------------

@st.cache_data(ttl=300)
def get_weather(lat, lon):
    """Fetch weather data with error handling"""
    weather = fetch_weather_data(lat, lon, OPENWEATHER_API_KEY)
    
    if weather is None:
        st.warning("⚠️ Unable to fetch weather data. Using default values.")
        return {
            "Temperature": 25.0,
            "Humidity": 60.0,
            "Wind Speed": 5.0,
            "Wind Direction": 180.0
        }
    
    return weather

@st.cache_data(ttl=300)
def get_pollution(lat, lon):
    """Fetch pollution data with error handling"""
    pollution = fetch_pollution_data(lat, lon, OPENWEATHER_API_KEY)
    
    if pollution is None:
        st.warning("⚠️ Unable to fetch pollution data. Using default values.")
        return {
            "pm25": 50.0,
            "pm10": 75.0,
            "no2": 40.0,
            "co": 500.0,
            "so2": 20.0,
            "o3": 60.0
        }
    
    return pollution

weather = get_weather(lat, lon)
pollution = get_pollution(lat, lon)

pm25 = pollution["pm25"]
pm10 = pollution["pm10"]
no2 = pollution["no2"]
co = pollution["co"]
so2 = pollution["so2"]
o3 = pollution["o3"]

# ------------------------------------------------
# CURRENT METRICS
# ------------------------------------------------

st.subheader("📈 Current Pollution Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("PM2.5", pm25)
col2.metric("PM10", pm10)
col3.metric("NO2", no2)
col4.metric("CO", co)

# ------------------------------------------------
# AQI STATUS
# ------------------------------------------------

aqi_status, aqi_color, aqi_emoji = get_aqi_status(pm25)
aqi_display = f"{aqi_status} {aqi_emoji}"

st.info(f"🌫 Air Quality Status: **{aqi_display}**")

# ------------------------------------------------
# PM2.5 GAUGE
# ------------------------------------------------

st.subheader("🌫 PM2.5 Air Quality Gauge")

fig_gauge = go.Figure(go.Indicator(

    mode="gauge+number",
    value=pm25,

    title={'text': "PM2.5 Level"},

    gauge={
        'axis': {'range': [0,300]},
        'steps': [
            {'range':[0,50],'color':"green"},
            {'range':[50,100],'color':"yellow"},
            {'range':[100,200],'color':"orange"},
            {'range':[200,300],'color':"red"}
        ]
    }
))

st.plotly_chart(fig_gauge,use_container_width=True)

# ------------------------------------------------
# POLLUTION TREND GRAPH
# ------------------------------------------------

st.subheader("📈 Pollution Trends")

city_data = df[df["city"]==city]

fig_trend = px.line(
    city_data,
    y=["pm25","pm10","no2","co"],
    title=f"Pollution Trend in {city}"
)

st.plotly_chart(fig_trend,use_container_width=True)

# ------------------------------------------------
# MODEL INPUT
# ------------------------------------------------

input_data = pd.DataFrame([{

    "co": co,
    "no2": no2,
    "o3": o3,
    "pm10": pm10,
    "pm25": pm25,
    "so2": so2,

    "Temperature": weather["Temperature"],
    "Humidity": weather["Humidity"],
    "Wind Speed": weather["Wind Speed"],
    "Wind Direction": weather["Wind Direction"],

    "dist_to_road": city_row["dist_to_road"],
    "dist_to_industry": city_row["dist_to_industry"],
    "dist_to_dump": city_row["dist_to_dump"],
    "dist_to_farmland": city_row["dist_to_farmland"]

}])

# ------------------------------------------------
# MODEL PREDICTION
# ------------------------------------------------

prediction = model.predict(input_data)
source = encoder.inverse_transform(prediction)

st.subheader("🔍 Predicted Pollution Source")
st.success(source[0])

# ------------------------------------------------
# TABS DASHBOARD
# ------------------------------------------------

st.markdown("## 📑 Environmental Data Dashboard")

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺 Pollution Map",
    "📊 Source Distribution",
    "🤖 AI Model Analysis",
    "📥 Download Report"
])

# ------------------------------------------------
# TAB 1 MAP
# ------------------------------------------------

with tab1:

    st.subheader("🗺 Pollution Map")

    m = folium.Map(location=[lat,lon],zoom_start=10)

    # folium.CircleMarker(
    #     location=[lat,lon],
    #     radius=12,
    #     popup=f"{city}\nPM2.5:{pm25}",
    #     color="red",
    #     fill=True
    # ).add_to(m)

    # st_folium(m,width=900,height=500)

    if POLLUTION_MAP_HTML.exists():
        with open(POLLUTION_MAP_HTML, "r", encoding="utf-8") as f:
            map_html = f.read()
        st.components.v1.html(map_html, height=650)
    else:
        st.warning(f"⚠️ Pollution map not found at {POLLUTION_MAP_HTML}")

# ------------------------------------------------
# TAB 2 SOURCE DISTRIBUTION
# ------------------------------------------------

with tab2:

    st.subheader("📊 Pollution Source Distribution")

    fig_pie = px.pie(
        df,
        names="pollution_source",
        hole=0.35
    )

    st.plotly_chart(fig_pie,use_container_width=True)

# ------------------------------------------------
# TAB 3 MODEL ANALYSIS
# ------------------------------------------------

with tab3:
    col1, col2 = st.columns(2)

    col1.markdown("### Confusion Matrix - XGBoost")
    if CONFUSION_MATRIX_XGBOOST.exists():
        col1.image(str(CONFUSION_MATRIX_XGBOOST), use_container_width=True)
    else:
        col1.warning("Confusion matrix image not found")

    col2.markdown("### Feature Importance")
    if CONFUSION_MATRIX_RF.exists():
        col2.image(str(CONFUSION_MATRIX_RF), use_container_width=True)
    else:
        col2.warning("Feature importance image not found")

# ------------------------------------------------
# TAB 4 DOWNLOAD REPORT
# ------------------------------------------------

with tab4:

    st.subheader("📥 Download Pollution Report")

    report_df = city_data.copy()

    report_df["Predicted Source"] = source[0]
    report_df["Generated Time"] = datetime.now()

    csv = report_df.to_csv(index=False)

    st.download_button(
        "Download CSV",
        csv,
        f"{city}_pollution_report.csv"
    )

    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        report_df.to_excel(writer,index=False)

    st.download_button(
        "Download Excel",
        excel_buffer,
        f"{city}_pollution_report.xlsx"
    )