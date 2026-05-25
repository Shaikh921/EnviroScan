"""
EnviroScan Dashboard - Enhanced Version
Real-Time Air Pollution Monitoring System with Advanced Features

New Features:
1. Real-Time Auto-Refresh
2. Multi-City Comparison
3. Historical Data Analysis
4. Dark Mode
5. Pollution Forecast
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import io
import logging
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression

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
    page_title="EnviroScan Dashboard - Enhanced",
    layout="wide",
    page_icon="🌍",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------
# DARK MODE TOGGLE
# ------------------------------------------------

# Initialize session state for dark mode
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Dark mode CSS
def apply_dark_mode():
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
        /* Main app background and text */
        .stApp {
            background-color: #0E1117 !important;
            color: #FAFAFA !important;
        }
        
        /* Sidebar styling */
        .stSidebar {
            background-color: #262730 !important;
        }
        
        /* All text elements */
        .stApp p, .stApp span, .stApp div, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
            color: #FAFAFA !important;
        }
        
        /* Metric containers */
        .stMetric {
            background-color: #262730 !important;
            padding: 10px !important;
            border-radius: 5px !important;
        }
        
        /* Metric labels and values */
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
            color: #FAFAFA !important;
        }
        
        /* Markdown text */
        .stMarkdown {
            color: #FAFAFA !important;
        }
        
        /* Dataframe text */
        .stDataFrame, .stDataFrame td, .stDataFrame th {
            color: #FAFAFA !important;
            background-color: #262730 !important;
        }
        
        /* Input fields */
        .stTextInput input, .stSelectbox select, .stMultiSelect {
            background-color: #262730 !important;
            color: #FAFAFA !important;
        }
        
        /* Buttons */
        .stButton button {
            background-color: #262730 !important;
            color: #FAFAFA !important;
        }
        
        /* Info/Warning/Success boxes */
        .stAlert {
            color: #0E1117 !important;
        }
        
        /* Tab labels */
        .stTabs [data-baseweb="tab"] {
            color: #FAFAFA !important;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            color: #FAFAFA !important;
        }
        
        /* Caption text */
        .caption {
            color: #B0B0B0 !important;
        }
        
        /* Radio button labels */
        .stRadio label {
            color: #FAFAFA !important;
        }
        
        /* Checkbox labels */
        .stCheckbox label {
            color: #FAFAFA !important;
        }
        
        /* Slider labels */
        .stSlider label {
            color: #FAFAFA !important;
        }
        </style>
        """, unsafe_allow_html=True)

apply_dark_mode()

# ------------------------------------------------
# TITLE AND HEADER
# ------------------------------------------------

st.title("🌍 EnviroScan: Enhanced Air Pollution Monitoring")
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
        
        # Convert datetime if exists
        if 'datetimeUtc' in df.columns:
            df['datetimeUtc'] = pd.to_datetime(df['datetimeUtc'])
        
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
# SIDEBAR CONTROLS
# ------------------------------------------------

st.sidebar.header("⚙️ Dashboard Controls")

# Dark Mode Toggle
dark_mode_toggle = st.sidebar.checkbox(
    "🌙 Dark Mode",
    value=st.session_state.dark_mode,
    key="dark_mode_checkbox"
)

if dark_mode_toggle != st.session_state.dark_mode:
    st.session_state.dark_mode = dark_mode_toggle
    st.rerun()

# Data Update Section
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Data Update")

if st.sidebar.button("🔄 Update Data from API", help="Fetch latest pollution data"):
    with st.spinner("Fetching latest data from API..."):
        try:
            # Import the updater
            import subprocess
            result = subprocess.run(
                ["python", "Dataset/Script/auto_data_updater.py", "--mode", "once"],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                st.sidebar.success("✅ Data updated successfully!")
                st.sidebar.info("Refresh the page to see new data")
                logger.info("Manual data update completed")
            else:
                st.sidebar.error("❌ Update failed. Check logs.")
                logger.error(f"Update error: {result.stderr}")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")
            logger.error(f"Manual update error: {e}")

st.sidebar.caption("💡 Updates all cities with latest API data")

# Auto-Refresh Toggle
st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Auto-Refresh")

auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=False)

if auto_refresh:
    refresh_interval = st.sidebar.slider(
        "Refresh Interval (seconds)",
        min_value=30,
        max_value=300,
        value=60,
        step=30
    )
    st.sidebar.info(f"⏱️ Auto-refreshing every {refresh_interval} seconds")
    
    # Use JavaScript-based meta refresh for smooth auto-refresh
    # This prevents the blur/blocking issue
    st.markdown(
        f"""
        <meta http-equiv="refresh" content="{refresh_interval}">
        <script>
            setTimeout(function(){{
                window.location.reload();
            }}, {refresh_interval * 1000});
        </script>
        """,
        unsafe_allow_html=True
    )
else:
    st.sidebar.caption("💡 Enable to automatically refresh data")

# ------------------------------------------------
# FEATURE SELECTION
# ------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.subheader("📊 View Selection")

view_mode = st.sidebar.radio(
    "Select View",
    ["Single City", "Multi-City Comparison", "Historical Analysis", "Forecast"],
    index=0
)

# ------------------------------------------------
# FETCH REAL-TIME DATA
# ------------------------------------------------

@st.cache_data(ttl=300)
def get_weather(lat, lon):
    """Fetch weather data with error handling"""
    weather = fetch_weather_data(lat, lon, OPENWEATHER_API_KEY)
    
    if weather is None:
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
        return {
            "pm25": 50.0,
            "pm10": 75.0,
            "no2": 40.0,
            "co": 500.0,
            "so2": 20.0,
            "o3": 60.0
        }
    
    return pollution

# ------------------------------------------------
# POLLUTION FORECAST FUNCTION
# ------------------------------------------------

def forecast_pollution(city_data, hours=24):
    """
    Forecast pollution levels using linear regression
    
    Args:
        city_data: Historical data for the city
        hours: Number of hours to forecast
        
    Returns:
        DataFrame with forecasted values
    """
    try:
        # Prepare data
        if 'datetimeUtc' not in city_data.columns:
            return None
        
        city_data = city_data.sort_values('datetimeUtc').copy()
        city_data['hour_index'] = range(len(city_data))
        
        # Train simple linear regression for PM2.5
        X = city_data[['hour_index']].values
        y = city_data['pm25'].values
        
        model_lr = LinearRegression()
        model_lr.fit(X, y)
        
        # Generate future hours
        last_index = city_data['hour_index'].max()
        future_indices = np.array([[last_index + i] for i in range(1, hours + 1)])
        
        # Predict
        predictions = model_lr.predict(future_indices)
        
        # Create forecast dataframe
        last_time = city_data['datetimeUtc'].max()
        future_times = [last_time + timedelta(hours=i) for i in range(1, hours + 1)]
        
        forecast_df = pd.DataFrame({
            'datetime': future_times,
            'pm25_forecast': np.maximum(predictions, 0)  # Ensure non-negative
        })
        
        return forecast_df
    
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        return None

# ================================================
# VIEW 1: SINGLE CITY VIEW
# ================================================

if view_mode == "Single City":
    
    city = st.sidebar.selectbox(
        "Select City",
        sorted(df["city"].unique())
    )
    
    city_row = df[df["city"] == city].iloc[0]
    lat = city_row["latitude"]
    lon = city_row["longitude"]
    
    # Fetch real-time data
    weather = get_weather(lat, lon)
    pollution = get_pollution(lat, lon)
    
    pm25 = pollution["pm25"]
    pm10 = pollution["pm10"]
    no2 = pollution["no2"]
    co = pollution["co"]
    so2 = pollution["so2"]
    o3 = pollution["o3"]
    
    # Current Metrics
    st.subheader(f"📈 Current Pollution Metrics - {city}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("PM2.5", f"{pm25:.1f}", help="Fine Particulate Matter")
    col2.metric("PM10", f"{pm10:.1f}", help="Coarse Particulate Matter")
    col3.metric("NO2", f"{no2:.1f}", help="Nitrogen Dioxide")
    col4.metric("CO", f"{co:.1f}", help="Carbon Monoxide")
    col5.metric("SO2", f"{so2:.1f}", help="Sulfur Dioxide")
    
    # AQI Status
    aqi_status, aqi_color, aqi_emoji = get_aqi_status(pm25)
    st.info(f"🌫 Air Quality Status: **{aqi_status} {aqi_emoji}**")
    
    # PM2.5 Gauge
    col_gauge, col_weather = st.columns([1, 1])
    
    with col_gauge:
        st.subheader("🌫 PM2.5 Air Quality Gauge")
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pm25,
            title={'text': "PM2.5 Level (μg/m³)"},
            gauge={
                'axis': {'range': [0, 300]},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 100], 'color': "yellow"},
                    {'range': [100, 200], 'color': "orange"},
                    {'range': [200, 300], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': pm25
                }
            }
        ))
        
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col_weather:
        st.subheader("🌤️ Weather Conditions")
        
        wcol1, wcol2 = st.columns(2)
        wcol1.metric("🌡️ Temperature", f"{weather['Temperature']:.1f}°C")
        wcol1.metric("💨 Wind Speed", f"{weather['Wind Speed']:.1f} m/s")
        wcol2.metric("💧 Humidity", f"{weather['Humidity']:.0f}%")
        wcol2.metric("🧭 Wind Direction", f"{weather['Wind Direction']:.0f}°")
    
    # Pollution Trends
    st.subheader("📈 Pollution Trends")
    
    city_data = df[df["city"] == city].copy()
    
    if 'datetimeUtc' in city_data.columns:
        city_data = city_data.sort_values('datetimeUtc')
        
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Scatter(
            x=city_data.index,
            y=city_data['pm25'],
            name='PM2.5',
            line=dict(color='red', width=2)
        ))
        
        fig_trend.add_trace(go.Scatter(
            x=city_data.index,
            y=city_data['pm10'],
            name='PM10',
            line=dict(color='orange', width=2)
        ))
        
        fig_trend.add_trace(go.Scatter(
            x=city_data.index,
            y=city_data['no2'],
            name='NO2',
            line=dict(color='blue', width=2)
        ))
        
        fig_trend.update_layout(
            title=f"Pollution Trends in {city}",
            xaxis_title="Time",
            yaxis_title="Concentration (μg/m³)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # Model Prediction
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
    
    prediction = model.predict(input_data)
    source = encoder.inverse_transform(prediction)
    
    st.subheader("🔍 Predicted Pollution Source")
    st.success(f"**{source[0]}**")

# ================================================
# VIEW 2: MULTI-CITY COMPARISON
# ================================================

elif view_mode == "Multi-City Comparison":
    
    st.subheader("🏙️ Multi-City Comparison")
    
    # City selection
    available_cities = sorted(df["city"].unique())
    
    selected_cities = st.sidebar.multiselect(
        "Select Cities to Compare (2-5)",
        available_cities,
        default=available_cities[:2] if len(available_cities) >= 2 else available_cities,
        help="Select 2 to 5 cities for comparison"
    )
    
    if len(selected_cities) == 0:
        st.info("👆 Please select cities from the sidebar to start comparison")
        st.stop()
    elif len(selected_cities) == 1:
        st.warning("⚠️ Please select at least one more city for comparison")
        st.stop()
    elif len(selected_cities) > 5:
        st.warning("⚠️ Maximum 5 cities allowed. Please deselect some cities for better visualization.")
        st.stop()
    else:
        # Fetch data for all selected cities
        comparison_data = []
        
        for city in selected_cities:
            city_row = df[df["city"] == city].iloc[0]
            lat = city_row["latitude"]
            lon = city_row["longitude"]
            
            pollution = get_pollution(lat, lon)
            weather = get_weather(lat, lon)
            
            comparison_data.append({
                'City': city,
                'PM2.5': pollution['pm25'],
                'PM10': pollution['pm10'],
                'NO2': pollution['no2'],
                'CO': pollution['co'],
                'SO2': pollution['so2'],
                'O3': pollution['o3'],
                'Temperature': weather['Temperature'],
                'Humidity': weather['Humidity']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display metrics
        st.markdown("### 📊 Current Pollution Levels")
        
        cols = st.columns(len(selected_cities))
        
        for idx, city in enumerate(selected_cities):
            city_data = comparison_df[comparison_df['City'] == city].iloc[0]
            pm25_val = city_data['PM2.5']
            aqi_status, aqi_color, aqi_emoji = get_aqi_status(pm25_val)
            
            with cols[idx]:
                st.markdown(f"#### {city}")
                st.metric("PM2.5", f"{pm25_val:.1f}")
                st.metric("PM10", f"{city_data['PM10']:.1f}")
                st.metric("NO2", f"{city_data['NO2']:.1f}")
                st.caption(f"AQI: {aqi_status} {aqi_emoji}")
        
        # Comparison Charts
        st.markdown("### 📊 Pollutant Comparison")
        
        # PM2.5 Comparison
        fig_pm25 = px.bar(
            comparison_df,
            x='City',
            y='PM2.5',
            title='PM2.5 Levels Comparison',
            color='PM2.5',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_pm25, use_container_width=True)
        
        # Multi-pollutant comparison
        pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
        
        fig_multi = go.Figure()
        
        for pollutant in pollutants:
            fig_multi.add_trace(go.Bar(
                name=pollutant,
                x=comparison_df['City'],
                y=comparison_df[pollutant]
            ))
        
        fig_multi.update_layout(
            title='All Pollutants Comparison',
            xaxis_title='City',
            yaxis_title='Concentration (μg/m³)',
            barmode='group'
        )
        
        st.plotly_chart(fig_multi, use_container_width=True)
        
        # Ranking
        st.markdown("### 🏆 City Rankings")
        
        ranking_df = comparison_df[['City', 'PM2.5']].sort_values('PM2.5')
        ranking_df['Rank'] = range(1, len(ranking_df) + 1)
        ranking_df['Status'] = ranking_df['PM2.5'].apply(lambda x: get_aqi_status(x)[0])
        
        st.dataframe(
            ranking_df[['Rank', 'City', 'PM2.5', 'Status']],
            use_container_width=True,
            hide_index=True
        )

# ================================================
# VIEW 3: HISTORICAL DATA ANALYSIS
# ================================================

elif view_mode == "Historical Analysis":
    
    st.subheader("📈 Historical Data Analysis")
    
    city = st.sidebar.selectbox(
        "Select City",
        sorted(df["city"].unique())
    )
    
    # Date range selector
    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 Time Period")
    
    time_range = st.sidebar.radio(
        "Select Range",
        ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time", "Custom"]
    )
    
    city_data = df[df["city"] == city].copy()
    
    # Check if we have data
    if len(city_data) == 0:
        st.error(f"No data available for {city}")
        st.stop()
    
    # Show info about the dataset
    if 'datetimeUtc' in city_data.columns:
        city_data = city_data.sort_values('datetimeUtc')
        
        # Get date range from actual data
        min_date = city_data['datetimeUtc'].min()
        max_date = city_data['datetimeUtc'].max()
        
        st.info(f"📅 Dataset covers: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        
        # Filter by time range based on actual data
        total_days = (max_date - min_date).days
        
        if time_range == "Last 7 Days" and total_days >= 7:
            cutoff = max_date - timedelta(days=7)
            city_data = city_data[city_data['datetimeUtc'] >= cutoff]
            st.caption(f"Showing last 7 days of data ({len(city_data)} records)")
        elif time_range == "Last 30 Days" and total_days >= 30:
            cutoff = max_date - timedelta(days=30)
            city_data = city_data[city_data['datetimeUtc'] >= cutoff]
            st.caption(f"Showing last 30 days of data ({len(city_data)} records)")
        elif time_range == "Last 90 Days" and total_days >= 90:
            cutoff = max_date - timedelta(days=90)
            city_data = city_data[city_data['datetimeUtc'] >= cutoff]
            st.caption(f"Showing last 90 days of data ({len(city_data)} records)")
        elif time_range == "Custom":
            # Use data's date range for custom selection
            start_date = st.sidebar.date_input(
                "Start Date", 
                value=min_date.date(),
                min_value=min_date.date(),
                max_value=max_date.date()
            )
            end_date = st.sidebar.date_input(
                "End Date", 
                value=max_date.date(),
                min_value=min_date.date(),
                max_value=max_date.date()
            )
            try:
                city_data = city_data[
                    (city_data['datetimeUtc'].dt.date >= start_date) &
                    (city_data['datetimeUtc'].dt.date <= end_date)
                ]
                st.caption(f"Custom range: {len(city_data)} records")
            except Exception as e:
                st.warning(f"Date filtering error: {e}. Showing all data.")
        else:
            st.caption(f"Showing all available data ({len(city_data)} records)")
    else:
        st.info("Showing all available data (datetime information not available)")
    
    # Check if we have valid data
    if len(city_data) == 0:
        st.warning(f"No data available for the selected time range")
        st.stop()
    
    # Summary Statistics
    st.markdown(f"### 📊 Summary Statistics - {city}")
    st.caption(f"Based on {len(city_data)} data points")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate statistics safely
    pm25_mean = city_data['pm25'].mean() if not city_data['pm25'].isna().all() else 0
    pm25_max = city_data['pm25'].max() if not city_data['pm25'].isna().all() else 0
    pm25_min = city_data['pm25'].min() if not city_data['pm25'].isna().all() else 0
    pm25_std = city_data['pm25'].std() if not city_data['pm25'].isna().all() else 0
    
    col1.metric("Avg PM2.5", f"{pm25_mean:.1f}")
    col2.metric("Max PM2.5", f"{pm25_max:.1f}")
    col3.metric("Min PM2.5", f"{pm25_min:.1f}")
    col4.metric("Std Dev", f"{pm25_std:.1f}")
    
    # Time Series Plot
    st.markdown("### 📈 Time Series Analysis")
    
    if 'datetimeUtc' in city_data.columns and len(city_data) > 0:
        fig_ts = go.Figure()
        
        # PM2.5 line
        fig_ts.add_trace(go.Scatter(
            x=city_data['datetimeUtc'],
            y=city_data['pm25'],
            mode='lines+markers',
            name='PM2.5',
            line=dict(color='#FF4B4B', width=2),
            marker=dict(size=4)
        ))
        
        # Add moving average if enough data
        window = min(24, len(city_data) // 4)  # Adaptive window size
        if len(city_data) >= window and window > 1:
            city_data['pm25_ma'] = city_data['pm25'].rolling(window=window, center=True).mean()
            
            fig_ts.add_trace(go.Scatter(
                x=city_data['datetimeUtc'],
                y=city_data['pm25_ma'],
                mode='lines',
                name=f'{window}-point Moving Avg',
                line=dict(color='#0068C9', width=3, dash='dash')
            ))
        
        # Add AQI threshold lines
        fig_ts.add_hline(y=50, line_dash="dot", line_color="green", 
                        annotation_text="Good", annotation_position="right")
        fig_ts.add_hline(y=100, line_dash="dot", line_color="yellow", 
                        annotation_text="Moderate", annotation_position="right")
        fig_ts.add_hline(y=200, line_dash="dot", line_color="orange", 
                        annotation_text="Poor", annotation_position="right")
        
        fig_ts.update_layout(
            title=f'PM2.5 Levels Over Time - {city}',
            xaxis_title='Date & Time',
            yaxis_title='PM2.5 Concentration (μg/m³)',
            hovermode='x unified',
            height=500,
            showlegend=True,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.warning("⚠️ Time series data not available")
    
    # Distribution Analysis
    st.markdown("### 📊 Distribution Analysis")
    
    col_hist, col_box = st.columns(2)
    
    with col_hist:
        # Histogram with better styling
        fig_hist = go.Figure()
        
        fig_hist.add_trace(go.Histogram(
            x=city_data['pm25'],
            nbinsx=30,
            name='PM2.5',
            marker=dict(
                color='#FF4B4B',
                line=dict(color='white', width=1)
            )
        ))
        
        fig_hist.update_layout(
            title='PM2.5 Distribution',
            xaxis_title='PM2.5 (μg/m³)',
            yaxis_title='Frequency',
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col_box:
        # Box plot with better styling
        fig_box = go.Figure()
        
        fig_box.add_trace(go.Box(
            y=city_data['pm25'],
            name='PM2.5',
            marker=dict(color='#0068C9'),
            boxmean='sd'  # Show mean and standard deviation
        ))
        
        fig_box.update_layout(
            title='PM2.5 Box Plot',
            yaxis_title='PM2.5 (μg/m³)',
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Correlation Analysis
    st.markdown("---")  # Visual separator
    st.markdown("### 🔗 Correlation Analysis")
    st.caption("Understanding relationships between different pollutants")
    
    # Check if all pollutant columns exist
    pollutants = ['pm25', 'pm10', 'no2', 'co', 'so2', 'o3']
    available_pollutants = [p for p in pollutants if p in city_data.columns]
    
    if len(available_pollutants) >= 2:
        try:
            # Calculate correlation
            corr_data = city_data[available_pollutants].corr()
            
            st.info(f"📊 Analyzing correlations between {len(available_pollutants)} pollutants")
            
            # Create heatmap with enhanced visibility
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_data.values,
                x=[p.upper() for p in corr_data.columns],
                y=[p.upper() for p in corr_data.columns],
                colorscale='RdBu_r',
                zmid=0,
                zmin=-1,
                zmax=1,
                text=corr_data.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 14, "color": "black"},
                colorbar=dict(
                    title="Correlation",
                    tickmode="linear",
                    tick0=-1,
                    dtick=0.5
                ),
                hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
            ))
            
            fig_corr.update_layout(
                title={
                    'text': 'Pollutant Correlation Heatmap',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'color': '#262730'}
                },
                xaxis_title='Pollutants',
                yaxis_title='Pollutants',
                height=600,
                width=800,
                template='plotly_white',
                font=dict(size=12)
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating correlation heatmap: {e}")
            logger.error(f"Correlation heatmap error: {e}")
        
        # Add interpretation
        st.markdown("---")
        st.markdown("#### 📝 Correlation Insights")
        st.caption("Key relationships discovered in the data")
        
        # Find strongest correlations
        corr_pairs = []
        for i in range(len(corr_data.columns)):
            for j in range(i+1, len(corr_data.columns)):
                corr_pairs.append({
                    'Pair': f"{corr_data.columns[i].upper()} ↔ {corr_data.columns[j].upper()}",
                    'Correlation': corr_data.iloc[i, j]
                })
        
        if corr_pairs:
            corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔴 Strongest Positive Correlations:**")
                st.caption("These pollutants tend to increase together")
                top_positive = corr_df.head(3)
                for idx, row in enumerate(top_positive.iterrows(), 1):
                    _, data = row
                    st.write(f"{idx}. {data['Pair']}: **{data['Correlation']:.3f}**")
            
            with col2:
                st.markdown("**🔵 Strongest Negative Correlations:**")
                st.caption("When one increases, the other tends to decrease")
                top_negative = corr_df.tail(3).sort_values('Correlation')
                for idx, row in enumerate(top_negative.iterrows(), 1):
                    _, data = row
                    st.write(f"{idx}. {data['Pair']}: **{data['Correlation']:.3f}**")
            
            # Add correlation strength guide
            st.markdown("---")
            st.markdown("**📚 Correlation Strength Guide:**")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.write("🟢 **Strong:** |r| > 0.7")
            with col_b:
                st.write("🟡 **Moderate:** 0.3 < |r| < 0.7")
            with col_c:
                st.write("⚪ **Weak:** |r| < 0.3")
    else:
        st.warning("⚠️ Insufficient pollutant data for correlation analysis")
    
    # Multi-Pollutant Time Series
    if 'datetimeUtc' in city_data.columns and len(available_pollutants) > 1:
        st.markdown("---")  # Visual separator
        st.markdown("### 📊 Multi-Pollutant Trends Comparison")
        st.caption("All pollutants normalized to 0-1 scale for easy comparison")
        
        fig_multi = go.Figure()
        
        colors = {
            'pm25': '#FF4B4B',
            'pm10': '#FFA500',
            'no2': '#0068C9',
            'co': '#00C9A7',
            'so2': '#C9007A',
            'o3': '#7A00C9'
        }
        
        for pollutant in available_pollutants:
            # Normalize for better visualization
            p_min = city_data[pollutant].min()
            p_max = city_data[pollutant].max()
            
            if p_max > p_min:  # Avoid division by zero
                normalized = (city_data[pollutant] - p_min) / (p_max - p_min)
            else:
                normalized = city_data[pollutant] * 0  # All zeros if constant
            
            fig_multi.add_trace(go.Scatter(
                x=city_data['datetimeUtc'],
                y=normalized,
                mode='lines',
                name=pollutant.upper(),
                line=dict(color=colors.get(pollutant, '#888888'), width=2.5),
                hovertemplate=f'<b>{pollutant.upper()}</b><br>Normalized: %{{y:.3f}}<br>Date: %{{x}}<extra></extra>'
            ))
        
        fig_multi.update_layout(
            title={
                'text': 'Normalized Pollutant Trends (0-1 Scale)',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title='Date & Time',
            yaxis_title='Normalized Value (0 = Min, 1 = Max)',
            height=500,
            hovermode='x unified',
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_multi, use_container_width=True)
        
        # Add explanation
        with st.expander("ℹ️ How to read this chart"):
            st.write("""
            **Normalization Explained:**
            - Each pollutant is scaled to 0-1 range
            - 0 = Minimum value observed for that pollutant
            - 1 = Maximum value observed for that pollutant
            - This allows comparing trends across pollutants with different units
            
            **What to look for:**
            - Pollutants moving together suggest common sources
            - Opposite movements suggest different emission patterns
            - Spikes indicate pollution events
            """)

# ================================================
# VIEW 4: POLLUTION FORECAST
# ================================================

if view_mode == "Forecast":
    
    st.subheader("🔮 Pollution Forecast")
    
    city = st.sidebar.selectbox(
        "Select City",
        sorted(df["city"].unique())
    )
    
    forecast_hours = st.sidebar.slider(
        "Forecast Hours",
        min_value=6,
        max_value=48,
        value=24,
        step=6
    )
    
    city_data = df[df["city"] == city].copy()
    
    # Generate forecast
    forecast_df = forecast_pollution(city_data, hours=forecast_hours)
    
    if forecast_df is not None:
        st.markdown(f"### 🔮 {forecast_hours}-Hour Forecast for {city}")
        
        # Current vs Forecast
        current_pm25 = city_data['pm25'].iloc[-1] if len(city_data) > 0 else 50
        forecast_avg = forecast_df['pm25_forecast'].mean()
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Current PM2.5", f"{current_pm25:.1f}")
        col2.metric("Forecast Avg", f"{forecast_avg:.1f}")
        col3.metric("Change", f"{forecast_avg - current_pm25:+.1f}")
        
        # Forecast Chart
        fig_forecast = go.Figure()
        
        # Historical data (last 24 hours)
        if 'datetimeUtc' in city_data.columns:
            recent_data = city_data.tail(24)
            
            fig_forecast.add_trace(go.Scatter(
                x=recent_data['datetimeUtc'],
                y=recent_data['pm25'],
                mode='lines',
                name='Historical',
                line=dict(color='blue', width=2)
            ))
        
        # Forecast data
        fig_forecast.add_trace(go.Scatter(
            x=forecast_df['datetime'],
            y=forecast_df['pm25_forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig_forecast.update_layout(
            title=f'{forecast_hours}-Hour PM2.5 Forecast',
            xaxis_title='Time',
            yaxis_title='PM2.5 (μg/m³)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Forecast Table
        st.markdown("### 📋 Detailed Forecast")
        
        forecast_display = forecast_df.copy()
        forecast_display['datetime'] = forecast_display['datetime'].dt.strftime('%Y-%m-%d %H:%M')
        forecast_display['pm25_forecast'] = forecast_display['pm25_forecast'].round(1)
        forecast_display['AQI Status'] = forecast_display['pm25_forecast'].apply(
            lambda x: get_aqi_status(x)[0]
        )
        
        st.dataframe(
            forecast_display.rename(columns={
                'datetime': 'Time',
                'pm25_forecast': 'PM2.5 Forecast'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Health Recommendations
        st.markdown("### 💊 Health Recommendations")
        
        max_forecast = forecast_df['pm25_forecast'].max()
        aqi_status, _, aqi_emoji = get_aqi_status(max_forecast)
        
        if aqi_status == "Good":
            st.success(f"{aqi_emoji} Air quality will be good. Normal outdoor activities recommended.")
        elif aqi_status == "Moderate":
            st.info(f"{aqi_emoji} Air quality will be moderate. Sensitive individuals should limit prolonged outdoor exertion.")
        elif aqi_status == "Poor":
            st.warning(f"{aqi_emoji} Air quality will be poor. Everyone should limit prolonged outdoor exertion.")
        elif aqi_status == "Very Poor":
            st.error(f"{aqi_emoji} Air quality will be very poor. Avoid outdoor activities. Use N95 masks if going outside.")
        else:
            st.error(f"{aqi_emoji} Air quality will be hazardous. Stay indoors. Use air purifiers.")
    
    else:
        st.warning("⚠️ Unable to generate forecast. Insufficient historical data.")

# ================================================
# ADDITIONAL TABS: MAP, SOURCE, AI MODEL, DOWNLOAD
# ================================================

st.markdown("---")
st.markdown("## 📑 Environmental Data Dashboard")

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺 Pollution Map",
    "📊 Source Distribution",
    "🤖 AI Model Analysis",
    "📥 Download Report"
])

# ------------------------------------------------
# TAB 1: POLLUTION MAP
# ------------------------------------------------

with tab1:
    st.subheader("🗺 Pollution Map")
    
    if POLLUTION_MAP_HTML.exists():
        with open(POLLUTION_MAP_HTML, "r", encoding="utf-8") as f:
            map_html = f.read()
        st.components.v1.html(map_html, height=650)
    else:
        st.warning(f"⚠️ Pollution map not found at {POLLUTION_MAP_HTML}")
        st.info("💡 Generate the map by running: `python Model_5_Geospatial/maps/generate_pollution_map.py`")

# ------------------------------------------------
# TAB 2: SOURCE DISTRIBUTION
# ------------------------------------------------

with tab2:
    st.subheader("📊 Pollution Source Distribution")
    
    if 'pollution_source' in df.columns:
        # Pie chart
        fig_pie = px.pie(
            df,
            names="pollution_source",
            title="Distribution of Pollution Sources",
            hole=0.35,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig_pie.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Bar chart
        source_counts = df['pollution_source'].value_counts().reset_index()
        source_counts.columns = ['Source', 'Count']
        
        fig_bar = px.bar(
            source_counts,
            x='Source',
            y='Count',
            title='Pollution Source Frequency',
            color='Count',
            color_continuous_scale='Reds'
        )
        
        fig_bar.update_layout(
            xaxis_title='Pollution Source',
            yaxis_title='Number of Occurrences',
            showlegend=False
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Statistics table
        st.markdown("### 📊 Source Statistics")
        st.dataframe(source_counts, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ Pollution source data not available in dataset")

# ------------------------------------------------
# TAB 3: AI MODEL ANALYSIS
# ------------------------------------------------

with tab3:
    st.subheader("🤖 AI Model Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Confusion Matrix - XGBoost")
        if CONFUSION_MATRIX_XGBOOST.exists():
            st.image(str(CONFUSION_MATRIX_XGBOOST), use_container_width=True)
        else:
            st.warning("⚠️ XGBoost confusion matrix not found")
            st.info("💡 Train the model by running: `python ModelScript/EnviroScan_Model.py`")
    
    with col2:
        st.markdown("### Confusion Matrix - Random Forest")
        if CONFUSION_MATRIX_RF.exists():
            st.image(str(CONFUSION_MATRIX_RF), use_container_width=True)
        else:
            st.warning("⚠️ Random Forest confusion matrix not found")
    
    # Model Information
    st.markdown("---")
    st.markdown("### 📋 Model Information")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.info("""
        **Model Type:**
        - XGBoost Classifier
        - Random Forest Classifier
        - Decision Tree Classifier
        """)
    
    with col_b:
        st.info("""
        **Features Used:**
        - Pollutant levels (PM2.5, PM10, NO2, CO, SO2, O3)
        - Weather data (Temp, Humidity, Wind)
        - Location features (distances)
        """)
    
    with col_c:
        st.info("""
        **Prediction Target:**
        - Pollution Source Classification
        - Categories: Industrial, Vehicular, Agricultural, etc.
        """)

# ------------------------------------------------
# TAB 4: DOWNLOAD REPORT
# ------------------------------------------------

with tab4:
    st.subheader("📥 Download Pollution Report")
    
    # Report type selection
    report_type = st.radio(
        "Select Report Type",
        ["Current City", "All Cities", "Custom Selection"],
        horizontal=True
    )
    
    if report_type == "Current City":
        # Get current city based on view mode
        if view_mode == "Single City" and 'city' in locals():
            report_df = df[df["city"] == city].copy()
            report_name = city
        else:
            st.info("👆 Please select a city from Single City view first")
            report_df = None
            report_name = "report"
    
    elif report_type == "All Cities":
        report_df = df.copy()
        report_name = "all_cities"
    
    else:  # Custom Selection
        selected_report_cities = st.multiselect(
            "Select Cities for Report",
            sorted(df["city"].unique()),
            default=[]
        )
        
        if selected_report_cities:
            report_df = df[df["city"].isin(selected_report_cities)].copy()
            report_name = "custom_selection"
        else:
            st.info("👆 Please select at least one city")
            report_df = None
            report_name = "report"
    
    if report_df is not None and len(report_df) > 0:
        # Remove timezone from datetime columns for Excel compatibility
        for col in report_df.columns:
            if pd.api.types.is_datetime64_any_dtype(report_df[col]):
                if report_df[col].dt.tz is not None:
                    report_df[col] = report_df[col].dt.tz_localize(None)
        
        # Add metadata
        report_df["Generated_Time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        report_df["Report_Type"] = report_type
        
        st.success(f"✅ Report ready: {len(report_df)} records")
        
        # Preview
        with st.expander("👁️ Preview Report Data"):
            st.dataframe(report_df.head(10), use_container_width=True)
        
        # Download buttons
        col_csv, col_excel = st.columns(2)
        
        with col_csv:
            csv = report_df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv,
                file_name=f"{report_name}_pollution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_excel:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                report_df.to_excel(writer, index=False, sheet_name="Pollution Data")
            
            st.download_button(
                label="📊 Download Excel",
                data=excel_buffer.getvalue(),
                file_name=f"{report_name}_pollution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        # Report statistics
        st.markdown("---")
        st.markdown("### 📊 Report Statistics")
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        stat_col1.metric("Total Records", len(report_df))
        stat_col2.metric("Cities Included", report_df['city'].nunique())
        stat_col3.metric("Avg PM2.5", f"{report_df['pm25'].mean():.1f}")
        stat_col4.metric("Max PM2.5", f"{report_df['pm25'].max():.1f}")

# ------------------------------------------------
# FOOTER
# ------------------------------------------------

st.markdown("---")
st.caption("EnviroScan Enhanced Dashboard | Data updated every 5 minutes | Powered by OpenWeather API")
