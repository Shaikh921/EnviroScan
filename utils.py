"""
EnviroScan Utility Functions
Common helper functions used across the project
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import requests
from sklearn.neighbors import BallTree

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================
# DATA LOADING
# ============================================

def load_dataset(file_path: Path, required_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load dataset with validation
    
    Args:
        file_path: Path to CSV file
        required_columns: List of required column names
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    logger.info(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path)
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Validate required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
    return df


# ============================================
# DATA CLEANING
# ============================================

def remove_missing_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """
    Handle missing values in dataset
    
    Args:
        df: Input DataFrame
        strategy: 'drop', 'interpolate', or 'fill'
        
    Returns:
        Cleaned DataFrame
    """
    initial_rows = len(df)
    missing_count = df.isnull().sum().sum()
    
    if missing_count == 0:
        logger.info("No missing values found")
        return df
    
    logger.info(f"Found {missing_count} missing values")
    
    if strategy == 'drop':
        df = df.dropna().reset_index(drop=True)
    elif strategy == 'interpolate':
        df = df.interpolate(method='linear').ffill().bfill()
    elif strategy == 'fill':
        df = df.fillna(df.mean())
    
    removed_rows = initial_rows - len(df)
    logger.info(f"Removed {removed_rows} rows with missing values")
    
    return df


def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for duplicates
        
    Returns:
        DataFrame without duplicates
    """
    initial_rows = len(df)
    df = df.drop_duplicates(subset=subset).reset_index(drop=True)
    removed_rows = initial_rows - len(df)
    
    if removed_rows > 0:
        logger.info(f"Removed {removed_rows} duplicate rows")
    
    return df


def remove_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr') -> pd.DataFrame:
    """
    Remove outliers from specified columns
    
    Args:
        df: Input DataFrame
        columns: Columns to check for outliers
        method: 'iqr' or 'zscore'
        
    Returns:
        DataFrame without outliers
    """
    initial_rows = len(df)
    
    if method == 'iqr':
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    
    elif method == 'zscore':
        from scipy import stats
        for col in columns:
            z_scores = np.abs(stats.zscore(df[col]))
            df = df[z_scores < 3]
    
    removed_rows = initial_rows - len(df)
    logger.info(f"Removed {removed_rows} outlier rows")
    
    return df


# ============================================
# GEOSPATIAL UTILITIES
# ============================================

def compute_distance(points_df: pd.DataFrame, 
                    feature_coords: np.ndarray, 
                    column_name: str) -> pd.DataFrame:
    """
    Compute haversine distance from points to nearest features
    
    Args:
        points_df: DataFrame with latitude and longitude columns
        feature_coords: Array of [lat, lon] coordinates
        column_name: Name for the distance column
        
    Returns:
        DataFrame with added distance column
    """
    if feature_coords.shape[0] == 0:
        points_df[column_name] = np.nan
        logger.warning(f"No features found for {column_name}")
        return points_df
    
    points_rad = np.radians(points_df[['latitude', 'longitude']].values)
    features_rad = np.radians(feature_coords)
    
    tree = BallTree(features_rad, metric='haversine')
    dist, _ = tree.query(points_rad, k=1)
    
    # Convert to meters (Earth radius = 6371 km)
    points_df[column_name] = dist.flatten() * 6371000
    
    logger.info(f"Computed {column_name} for {len(points_df)} points")
    
    return points_df


# ============================================
# API UTILITIES
# ============================================

def fetch_weather_data(lat: float, lon: float, api_key: str, 
                      retries: int = 3) -> Optional[Dict]:
    """
    Fetch current weather data from OpenWeather API
    
    Args:
        lat: Latitude
        lon: Longitude
        api_key: OpenWeather API key
        retries: Number of retry attempts
        
    Returns:
        Weather data dictionary or None if failed
    """
    url = f"https://api.openweathermap.org/data/2.5/weather"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': api_key,
        'units': 'metric'
    }
    
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                "Temperature": data["main"]["temp"],
                "Humidity": data["main"]["humidity"],
                "Wind Speed": data["wind"]["speed"],
                "Wind Direction": data["wind"]["deg"]
            }
        
        except requests.exceptions.RequestException as e:
            logger.warning(f"Weather API attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                logger.error("Weather API failed after all retries")
                return None
    
    return None


def fetch_pollution_data(lat: float, lon: float, api_key: str, 
                        retries: int = 3) -> Optional[Dict]:
    """
    Fetch current pollution data from OpenWeather API
    
    Args:
        lat: Latitude
        lon: Longitude
        api_key: OpenWeather API key
        retries: Number of retry attempts
        
    Returns:
        Pollution data dictionary or None if failed
    """
    url = f"https://api.openweathermap.org/data/2.5/air_pollution"
    params = {
        'lat': lat,
        'lon': lon,
        'appid': api_key
    }
    
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            comp = data["list"][0]["components"]
            
            return {
                "pm25": comp.get("pm2_5", 0),
                "pm10": comp.get("pm10", 0),
                "no2": comp.get("no2", 0),
                "co": comp.get("co", 0),
                "so2": comp.get("so2", 0),
                "o3": comp.get("o3", 0)
            }
        
        except requests.exceptions.RequestException as e:
            logger.warning(f"Pollution API attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                logger.error("Pollution API failed after all retries")
                return None
    
    return None


# ============================================
# MODEL UTILITIES
# ============================================

def validate_input_data(data: pd.DataFrame, required_features: List[str]) -> bool:
    """
    Validate input data for model prediction
    
    Args:
        data: Input DataFrame
        required_features: List of required feature names
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    missing_features = set(required_features) - set(data.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Check for NaN values
    if data[required_features].isnull().any().any():
        raise ValueError("Input data contains NaN values")
    
    # Check for infinite values
    if np.isinf(data[required_features].values).any():
        raise ValueError("Input data contains infinite values")
    
    return True


def add_noise_to_features(X: pd.DataFrame, noise_level: float = 0.15) -> pd.DataFrame:
    """
    Add controlled Gaussian noise to features
    
    Args:
        X: Feature DataFrame
        noise_level: Standard deviation multiplier for noise
        
    Returns:
        DataFrame with added noise
    """
    X_noisy = X.copy()
    
    for col in X.columns:
        std_dev = X[col].std()
        noise = np.random.normal(0, noise_level * std_dev, X.shape[0])
        X_noisy[col] += noise
    
    logger.info(f"Added {noise_level} noise level to features")
    
    return X_noisy


# ============================================
# VISUALIZATION UTILITIES
# ============================================

def get_source_color(source: str) -> str:
    """
    Get color for pollution source
    
    Args:
        source: Pollution source name
        
    Returns:
        Color name
    """
    colors = {
        "Industrial": "red",
        "Vehicular": "blue",
        "Agricultural": "green",
        "Burning": "orange",
        "Natural": "purple"
    }
    return colors.get(source, "gray")


# ============================================
# FILE UTILITIES
# ============================================

def save_dataframe(df: pd.DataFrame, file_path: Path, index: bool = False):
    """
    Save DataFrame to CSV with logging
    
    Args:
        df: DataFrame to save
        file_path: Output file path
        index: Whether to save index
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=index)
    logger.info(f"Saved dataset to {file_path}. Shape: {df.shape}")


def create_directory_structure(base_path: Path):
    """
    Create project directory structure
    
    Args:
        base_path: Base project path
    """
    directories = [
        base_path / "Dataset",
        base_path / "Dataset" / "city_pollution",
        base_path / "Dataset" / "Script",
        base_path / "Models",
        base_path / "Images" / "Dashboard",
        base_path / "ModelScript" / "UniqueScript",
        base_path / "Model_5_Geospatial" / "maps",
        base_path / "Model_5_Geospatial" / "html_exports",
        base_path / "Model_6_Dashboard"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    logger.info("Directory structure created successfully")


if __name__ == "__main__":
    # Test utilities
    logger.info("EnviroScan utilities loaded successfully")
