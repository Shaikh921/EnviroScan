import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import requests
import io
import os
from datetime import datetime
import folium
from streamlit_folium import st_folium
from dotenv import load_dotenv
load_dotenv()

# ------------------------------------------------
# API KEY
# ------------------------------------------------

OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="EnviroScan Dashboard",
    layout="wide"
)

st.title("🌍 EnviroScan: Real-Time Air Pollution Monitoring System")
st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ------------------------------------------------
# LOAD DATASET
# ------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(
        "C:/Infosys/Environ_Scan_Project/Dataset/Final_Dataset_Labeled_Balanced.csv"
    )

    df.columns = df.columns.str.strip()

    return df


df = load_data()

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load(
        "C:/Infosys/Environ_Scan_Project/Models/XGBoost.joblib"
    )

    encoder = joblib.load(
        "C:/Infosys/Environ_Scan_Project/Models/LabelEncoder.joblib"
    )

    return model, encoder


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
# WEATHER API
# ------------------------------------------------

@st.cache_data(ttl=300)
def get_weather(lat, lon):

    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}&units=metric"

    try:
        data = requests.get(url).json()

        weather = {
            "Temperature": data["main"]["temp"],
            "Humidity": data["main"]["humidity"],
            "Wind Speed": data["wind"]["speed"],
            "Wind Direction": data["wind"]["deg"]
        }

    except:

        weather = {
            "Temperature":0,
            "Humidity":0,
            "Wind Speed":0,
            "Wind Direction":0
        }

    return weather


# ------------------------------------------------
# POLLUTION API
# ------------------------------------------------

@st.cache_data(ttl=300)
def get_pollution(lat, lon):

    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}"

    try:

        data = requests.get(url).json()

        comp = data["list"][0]["components"]

        pollution = {
            "pm25": comp["pm2_5"],
            "pm10": comp["pm10"],
            "no2": comp["no2"],
            "co": comp["co"],
            "so2": comp["so2"],
            "o3": comp["o3"]
        }

    except:

        pollution = {
            "pm25":0,
            "pm10":0,
            "no2":0,
            "co":0,
            "so2":0,
            "o3":0
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

def get_aqi_status(pm25):

    if pm25 <= 50:
        return "Good 🟢"
    elif pm25 <= 100:
        return "Moderate 🟡"
    elif pm25 <= 200:
        return "Poor 🟠"
    elif pm25 <= 300:
        return "Very Poor 🔴"
    else:
        return "Hazardous 🟣"

aqi = get_aqi_status(pm25)

st.info(f"🌫 Air Quality Status: **{aqi}**")

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

    map_path="C:/Infosys/Environ_Scan_Project/Model_5_Geospatial/html_exports/pollution_map.html"

    if os.path.exists(map_path):

        with open(map_path,"r",encoding="utf-8") as f:
            map_html=f.read()

        st.components.v1.html(map_html,height=650)

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

    col1,col2 = st.columns(2)

    matrix_path="C:/Infosys/Environ_Scan_Project/Images/Matrix-XGBoost.png"
    feature_path="C:/Infosys/Environ_Scan_Project/Images/Random_forest.png"

    col1.markdown("### Confusion Matrix - XGBoost")

    if os.path.exists(matrix_path):
        col1.image(matrix_path,use_container_width=True)

    col2.markdown("### Feature Importance")

    if os.path.exists(feature_path):
        col2.image(feature_path,use_container_width=True)

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