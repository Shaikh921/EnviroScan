"""
EnviroScan Configuration File
Centralized configuration for all project settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================
# PROJECT PATHS
# ============================================

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "Dataset"
MODELS_DIR = PROJECT_ROOT / "Models"
IMAGES_DIR = PROJECT_ROOT / "Images"
GEOSPATIAL_DIR = PROJECT_ROOT / "Model_5_Geospatial"

# Dataset paths
MAIN_POLLUTION_DATASET = DATA_DIR / "Main_Pollution_Dataset.csv"
POLLUTION_WEATHER_DATASET = DATA_DIR / "Pollution_Weather_Dataset.csv"
LOCATION_FEATURES_DATASET = DATA_DIR / "Location_Features_Dataset.csv"
FINAL_DATASET_CLEANED = DATA_DIR / "Final_Dataset_Cleaned.csv"
FINAL_DATASET_LABELED = DATA_DIR / "Final_Dataset_Labeled.csv"
FINAL_DATASET_BALANCED = DATA_DIR / "Final_Dataset_Labeled_Balanced.csv"
FINAL_PREDICTIONS = DATA_DIR / "Final_Predictions.csv"

# Model paths
RANDOM_FOREST_MODEL = MODELS_DIR / "RandomForest.joblib"
DECISION_TREE_MODEL = MODELS_DIR / "DecisionTree.joblib"
XGBOOST_MODEL = MODELS_DIR / "XGBoost.joblib"
LABEL_ENCODER = MODELS_DIR / "LabelEncoder.joblib"

# Geospatial paths
POLLUTION_MAP_HTML = GEOSPATIAL_DIR / "html_exports" / "pollution_map.html"

# Image paths
CONFUSION_MATRIX_RF = IMAGES_DIR / "Random_forest.png"
CONFUSION_MATRIX_XGBOOST = IMAGES_DIR / "Matrix-XGBoost.png"

# ============================================
# API CONFIGURATION
# ============================================

# Check environment variable first, then fallback to Streamlit secrets
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_KEY")
if not OPENWEATHER_API_KEY or OPENWEATHER_API_KEY == "your_api_key_here":
    try:
        import streamlit as st
        if "OPENWEATHER_KEY" in st.secrets:
            OPENWEATHER_API_KEY = st.secrets["OPENWEATHER_KEY"]
    except Exception:
        pass

OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"
OPENWEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# ============================================
# MODEL CONFIGURATION
# ============================================

# Feature columns
POLLUTION_FEATURES = ['co', 'no2', 'o3', 'pm10', 'pm25', 'so2']
WEATHER_FEATURES = ['Temperature', 'Humidity', 'Wind Speed', 'Wind Direction']
LOCATION_FEATURES = ['dist_to_road', 'dist_to_industry', 'dist_to_dump', 'dist_to_farmland']
ALL_FEATURES = POLLUTION_FEATURES + WEATHER_FEATURES + LOCATION_FEATURES

# Target variable
TARGET_COLUMN = 'pollution_source'

# Pollution source classes
POLLUTION_SOURCES = ['Agricultural', 'Burning', 'Industrial', 'Natural', 'Vehicular']

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
NOISE_LEVEL = 0.15

# ============================================
# DATA LABELING THRESHOLDS
# ============================================

# Percentile thresholds for pollutants
POLLUTANT_PERCENTILE = 0.65

# Distance thresholds (meters)
ROAD_PROXIMITY_THRESHOLD = 600
INDUSTRY_PROXIMITY_THRESHOLD = 2000
DUMP_PROXIMITY_THRESHOLD = 6000
FARMLAND_PROXIMITY_THRESHOLD = 4000

# Weather thresholds
HUMIDITY_THRESHOLD = 70

# ============================================
# AQI THRESHOLDS (PM2.5 based)
# ============================================

AQI_THRESHOLDS = {
    'Good': (0, 50),
    'Moderate': (51, 100),
    'Poor': (101, 200),
    'Very Poor': (201, 300),
    'Hazardous': (301, float('inf'))
}

AQI_COLORS = {
    'Good': 'green',
    'Moderate': 'yellow',
    'Poor': 'orange',
    'Very Poor': 'red',
    'Hazardous': 'purple'
}

AQI_EMOJIS = {
    'Good': '🟢',
    'Moderate': '🟡',
    'Poor': '🟠',
    'Very Poor': '🔴',
    'Hazardous': '🟣'
}

# ============================================
# MAP CONFIGURATION
# ============================================

# India center coordinates
INDIA_CENTER = [20.5937, 78.9629]
DEFAULT_ZOOM = 5

# Source colors for map markers
SOURCE_COLORS = {
    "Industrial": "red",
    "Vehicular": "blue",
    "Agricultural": "green",
    "Burning": "orange",
    "Natural": "purple"
}

# ============================================
# HYPERPARAMETER GRIDS
# ============================================

# Random Forest
RF_PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'min_samples_split': [8, 12],
    'min_samples_leaf': [5, 8]
}

# Decision Tree
DT_PARAM_GRID = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [5, 10]
}

# XGBoost
XGB_PARAM_DIST = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# ============================================
# VALIDATION
# ============================================

def validate_config():
    """Validate that all required configurations are set"""
    errors = []
    
    if not OPENWEATHER_API_KEY:
        errors.append("OPENWEATHER_KEY not found in environment variables")
    
    if not DATA_DIR.exists():
        errors.append(f"Data directory not found: {DATA_DIR}")
    
    if not MODELS_DIR.exists():
        errors.append(f"Models directory not found: {MODELS_DIR}")
    
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_aqi_status(pm25_value):
    """Get AQI status based on PM2.5 value"""
    for status, (min_val, max_val) in AQI_THRESHOLDS.items():
        if min_val <= pm25_value <= max_val:
            return status, AQI_COLORS[status], AQI_EMOJIS[status]
    return 'Unknown', 'gray', '⚪'

def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        DATA_DIR,
        MODELS_DIR,
        IMAGES_DIR,
        GEOSPATIAL_DIR / "html_exports",
        GEOSPATIAL_DIR / "maps"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Test configuration
    try:
        validate_config()
        print("✅ Configuration validated successfully")
        print(f"📁 Project Root: {PROJECT_ROOT}")
        print(f"🔑 API Key: {'Set' if OPENWEATHER_API_KEY else 'Not Set'}")
    except ValueError as e:
        print(f"❌ Configuration Error:\n{e}")
