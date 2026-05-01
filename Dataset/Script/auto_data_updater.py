"""
EnviroScan - Automated Data Updater
Fetches real-time pollution data from API and updates city-wise datasets

Features:
- City-wise data collection
- Automatic scheduling (every 6 hours)
- Manual update option
- Data validation and cleaning
- Appends to existing datasets
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import OPENWEATHER_API_KEY, DATA_DIR, FINAL_DATASET_BALANCED
from utils import fetch_weather_data, fetch_pollution_data, logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ============================================
# CONFIGURATION
# ============================================

# City data with coordinates
CITIES = {
    "Fort William, Kolkata - WBPCB": {"lat": 22.55664, "lon": 88.342674},
    "Nehru Nagar, Delhi - DPCC": {"lat": 28.5672, "lon": 77.2537},
    "Karni Colony, Nagpur - MPCB": {"lat": 21.1458, "lon": 79.0882},
    "Jawahar Nagar, Amritsar - PPCB": {"lat": 31.6340, "lon": 74.8723},
    "Hebbal 1st Stage, Bengaluru - KSPCB": {"lat": 13.0358, "lon": 77.5970},
    # Add more cities as needed
}

# Update interval (in hours)
UPDATE_INTERVAL_HOURS = 6

# ============================================
# DATA COLLECTION FUNCTIONS
# ============================================

def fetch_city_data(city_name, lat, lon):
    """
    Fetch current pollution and weather data for a city
    
    Args:
        city_name: Name of the city
        lat: Latitude
        lon: Longitude
        
    Returns:
        Dictionary with pollution and weather data
    """
    try:
        # Fetch pollution data
        pollution = fetch_pollution_data(lat, lon, OPENWEATHER_API_KEY)
        
        if pollution is None:
            logger.warning(f"Failed to fetch pollution data for {city_name}")
            return None
        
        # Fetch weather data
        weather = fetch_weather_data(lat, lon, OPENWEATHER_API_KEY)
        
        if weather is None:
            logger.warning(f"Failed to fetch weather data for {city_name}")
            return None
        
        # Combine data
        data = {
            'city': city_name,
            'latitude': lat,
            'longitude': lon,
            'datetimeUtc': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S+00:00'),
            'co': pollution['co'],
            'no2': pollution['no2'],
            'o3': pollution['o3'],
            'pm10': pollution['pm10'],
            'pm25': pollution['pm25'],
            'so2': pollution['so2'],
            'Temperature': weather['Temperature'],
            'Humidity': weather['Humidity'],
            'Wind Speed': weather['Wind Speed'],
            'Wind Direction': weather['Wind Direction']
        }
        
        logger.info(f"✅ Successfully fetched data for {city_name}")
        return data
    
    except Exception as e:
        logger.error(f"Error fetching data for {city_name}: {e}")
        return None


def update_city_dataset(city_name, new_data):
    """
    Update individual city CSV file with new data
    
    Args:
        city_name: Name of the city
        new_data: Dictionary with new pollution data
        
    Returns:
        Boolean indicating success
    """
    try:
        # Create city pollution folder if not exists
        city_folder = DATA_DIR / "city_pollution"
        city_folder.mkdir(parents=True, exist_ok=True)
        
        # Clean city name for filename
        filename = city_name.split(',')[0].replace(' ', '_') + '.csv'
        file_path = city_folder / filename
        
        # Convert to DataFrame
        new_row = pd.DataFrame([new_data])
        
        # Check if file exists
        if file_path.exists():
            # Read existing data
            existing_df = pd.read_csv(file_path)
            
            # Append new data
            updated_df = pd.concat([existing_df, new_row], ignore_index=True)
            
            # Remove duplicates based on datetime
            updated_df = updated_df.drop_duplicates(subset=['datetimeUtc'], keep='last')
            
            # Save
            updated_df.to_csv(file_path, index=False)
            logger.info(f"✅ Updated {filename} - Total records: {len(updated_df)}")
        else:
            # Create new file
            new_row.to_csv(file_path, index=False)
            logger.info(f"✅ Created new file {filename}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error updating dataset for {city_name}: {e}")
        return False


def update_main_dataset(city_data_list):
    """
    Update the main consolidated dataset
    
    Args:
        city_data_list: List of dictionaries with city data
        
    Returns:
        Boolean indicating success
    """
    try:
        # Convert to DataFrame
        new_data = pd.DataFrame(city_data_list)
        
        # Check if main dataset exists
        if FINAL_DATASET_BALANCED.exists():
            # Read existing data
            existing_df = pd.read_csv(FINAL_DATASET_BALANCED)
            
            # Get location features for new data
            # For simplicity, use average values from existing data
            for city in new_data['city'].unique():
                city_existing = existing_df[existing_df['city'] == city]
                
                if len(city_existing) > 0:
                    new_data.loc[new_data['city'] == city, 'dist_to_road'] = city_existing['dist_to_road'].iloc[0]
                    new_data.loc[new_data['city'] == city, 'dist_to_industry'] = city_existing['dist_to_industry'].iloc[0]
                    new_data.loc[new_data['city'] == city, 'dist_to_dump'] = city_existing['dist_to_dump'].iloc[0]
                    new_data.loc[new_data['city'] == city, 'dist_to_farmland'] = city_existing['dist_to_farmland'].iloc[0]
            
            # Append new data
            updated_df = pd.concat([existing_df, new_data], ignore_index=True)
            
            # Remove duplicates
            updated_df = updated_df.drop_duplicates(subset=['city', 'datetimeUtc'], keep='last')
            
            # Save
            updated_df.to_csv(FINAL_DATASET_BALANCED, index=False)
            logger.info(f"✅ Updated main dataset - Total records: {len(updated_df)}")
            
            return True
        else:
            logger.warning("Main dataset not found. Only city files updated.")
            return False
    
    except Exception as e:
        logger.error(f"Error updating main dataset: {e}")
        return False


def collect_all_cities():
    """
    Collect data for all cities
    
    Returns:
        List of successfully collected city data
    """
    logger.info("=" * 60)
    logger.info("Starting data collection for all cities")
    logger.info("=" * 60)
    
    collected_data = []
    success_count = 0
    fail_count = 0
    
    for city_name, coords in CITIES.items():
        logger.info(f"Fetching data for: {city_name}")
        
        # Fetch data
        data = fetch_city_data(city_name, coords['lat'], coords['lon'])
        
        if data:
            # Update city dataset
            if update_city_dataset(city_name, data):
                collected_data.append(data)
                success_count += 1
            else:
                fail_count += 1
        else:
            fail_count += 1
        
        # Small delay to avoid rate limiting
        time.sleep(1)
    
    logger.info("=" * 60)
    logger.info(f"Data collection completed!")
    logger.info(f"✅ Success: {success_count} cities")
    logger.info(f"❌ Failed: {fail_count} cities")
    logger.info("=" * 60)
    
    return collected_data


def run_update_cycle():
    """
    Run one complete update cycle
    """
    start_time = datetime.now()
    logger.info(f"🚀 Starting update cycle at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Collect data for all cities
    collected_data = collect_all_cities()
    
    # Update main dataset if we have data
    if collected_data:
        update_main_dataset(collected_data)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info(f"✅ Update cycle completed in {duration:.2f} seconds")
    logger.info(f"Next update in {UPDATE_INTERVAL_HOURS} hours")


def run_continuous():
    """
    Run continuous updates every UPDATE_INTERVAL_HOURS
    """
    logger.info("🔄 Starting continuous data updater")
    logger.info(f"Update interval: {UPDATE_INTERVAL_HOURS} hours")
    
    while True:
        try:
            run_update_cycle()
            
            # Wait for next update
            sleep_seconds = UPDATE_INTERVAL_HOURS * 3600
            logger.info(f"💤 Sleeping for {UPDATE_INTERVAL_HOURS} hours...")
            time.sleep(sleep_seconds)
        
        except KeyboardInterrupt:
            logger.info("⏹️ Stopping data updater (Ctrl+C pressed)")
            break
        except Exception as e:
            logger.error(f"Error in update cycle: {e}")
            logger.info("Retrying in 5 minutes...")
            time.sleep(300)


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='EnviroScan Data Updater')
    parser.add_argument(
        '--mode',
        choices=['once', 'continuous'],
        default='once',
        help='Run once or continuously'
    )
    parser.add_argument(
        '--city',
        type=str,
        help='Update specific city only'
    )
    
    args = parser.parse_args()
    
    if args.city:
        # Update specific city
        if args.city in CITIES:
            coords = CITIES[args.city]
            data = fetch_city_data(args.city, coords['lat'], coords['lon'])
            if data:
                update_city_dataset(args.city, data)
        else:
            logger.error(f"City '{args.city}' not found in configuration")
    
    elif args.mode == 'once':
        # Run once
        run_update_cycle()
    
    else:
        # Run continuously
        run_continuous()
