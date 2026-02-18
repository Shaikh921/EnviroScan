import pandas as pd
import requests
from tqdm import tqdm

# =====================================================
# 1️⃣ LOAD POLLUTION DATA
# =====================================================

pollution_df = pd.read_csv("Main_Pollution_Dataset.csv")

# Convert to UTC datetime
pollution_df['datetimeUtc'] = pd.to_datetime(
    pollution_df['datetimeUtc'],
    utc=True
)

# Create hourly key (because pollution is 15-min data)
pollution_df['datetime_hour'] = pollution_df['datetimeUtc'].dt.floor('H')

# Extract date for API call
pollution_df['date'] = pollution_df['datetime_hour'].dt.date

# =====================================================
# 2️⃣ PREPARE UNIQUE LOCATION + DATE COMBINATIONS
# =====================================================

unique_locations = pollution_df[
    ['city', 'latitude', 'longitude', 'date']
].drop_duplicates()

weather_data = []

# =====================================================
# 3️⃣ FETCH WEATHER DATA (UTC TIMEZONE!)
# =====================================================

for _, row in tqdm(unique_locations.iterrows(), total=len(unique_locations)):

    city = row['city']
    lat = row['latitude']
    lon = row['longitude']
    date = row['date']

    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={date}&end_date={date}"
        f"&hourly=temperature_2m,relative_humidity_2m,"
        f"wind_speed_10m,wind_direction_10m"
        f"&timezone=UTC"   
        
    )

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        hourly = data.get("hourly", {})

        if "time" in hourly:
            for i in range(len(hourly["time"])):
                weather_data.append({
                    "city": city,
                    "latitude": lat,
                    "longitude": lon,
                    "datetime_hour": pd.to_datetime(hourly["time"][i], utc=True),
                    "Temperature": hourly["temperature_2m"][i],
                    "Humidity": hourly["relative_humidity_2m"][i],
                    "Wind Speed": hourly["wind_speed_10m"][i],
                    "Wind Direction": hourly["wind_direction_10m"][i]
                })

# =====================================================
# 4️⃣ CREATE WEATHER DATAFRAME
# =====================================================

weather_df = pd.DataFrame(weather_data)

# Remove duplicates if any
weather_df = weather_df.drop_duplicates(
    subset=['city', 'latitude', 'longitude', 'datetime_hour']
)

# =====================================================
# 5️⃣ MERGE POLLUTION + WEATHER
# =====================================================

final_df = pd.merge(
    pollution_df,
    weather_df,
    on=['city', 'latitude', 'longitude', 'datetime_hour'],
    how='left'
)

# =====================================================
# 6️⃣ CLEANUP (Optional)
# =====================================================

# Drop helper columns if not needed
final_df.drop(columns=['date'], inplace=True)

# =====================================================
# 7️⃣ SAVE FINAL DATASET
# =====================================================

final_df.to_csv("Final_Pollution_Weather_Dataset.csv", index=False)

print("✅ Final merged dataset created successfully!")
print("Total rows:", len(final_df))
print("Missing weather rows:", final_df['Temperature'].isna().sum())
