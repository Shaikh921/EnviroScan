import osmnx as ox
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

print("Loading dataset...")

df = pd.read_csv("Pollution_Weather_Dataset.csv")
locations = df[['city', 'latitude', 'longitude']].drop_duplicates()

all_results = []

# -------------------------------------------------
# Fast Haversine Distance
# -------------------------------------------------
def compute_distance(points_df, feature_coords, column_name):

    if feature_coords.shape[0] == 0:
        points_df[column_name] = np.nan
        return points_df

    points_rad = np.radians(points_df[['latitude', 'longitude']].values)
    features_rad = np.radians(feature_coords)

    tree = BallTree(features_rad, metric='haversine')
    dist, _ = tree.query(points_rad, k=1)

    points_df[column_name] = dist.flatten() * 6371000  # meters

    return points_df


cities = locations['city'].unique()
print(f"Processing {len(cities)} cities...\n")

for city in cities:

    print(f"Processing city: {city}")

    city_data = locations[locations['city'] == city].copy()

    center_lat = city_data['latitude'].mean()
    center_lon = city_data['longitude'].mean()

    tags = {
        "highway": ["primary", "secondary"],
        "landuse": ["industrial", "farmland"],
        "amenity": "landfill"
    }

    try:
        osm = ox.features_from_point(
            (center_lat, center_lon),
            tags=tags,
            dist=20000
        )
    except Exception as e:
        print(f"OSM fetch failed for {city}: {e}")
        continue

    if osm.empty:
        print("No OSM features found.")
        continue

    # Extract coordinates safely
    osm["lat"] = osm.geometry.centroid.y
    osm["lon"] = osm.geometry.centroid.x

    # ------------------------------
    # Safe Column Filtering
    # ------------------------------

    roads = osm[osm['highway'].notna()] if 'highway' in osm.columns else pd.DataFrame()
    industrial = osm[osm['landuse'] == "industrial"] if 'landuse' in osm.columns else pd.DataFrame()
    farmland = osm[osm['landuse'] == "farmland"] if 'landuse' in osm.columns else pd.DataFrame()
    landfill = osm[osm['amenity'] == "landfill"] if 'amenity' in osm.columns else pd.DataFrame()

    # ------------------------------
    # Distance Computation
    # ------------------------------

    city_data = compute_distance(
        city_data,
        roads[['lat', 'lon']].values if not roads.empty else np.empty((0, 2)),
        "dist_to_road"
    )

    city_data = compute_distance(
        city_data,
        industrial[['lat', 'lon']].values if not industrial.empty else np.empty((0, 2)),
        "dist_to_industry"
    )

    city_data = compute_distance(
        city_data,
        landfill[['lat', 'lon']].values if not landfill.empty else np.empty((0, 2)),
        "dist_to_dump"
    )

    city_data = compute_distance(
        city_data,
        farmland[['lat', 'lon']].values if not farmland.empty else np.empty((0, 2)),
        "dist_to_farmland"
    )

    all_results.append(city_data)

print("\nSaving dataset...")

final_location_df = pd.concat(all_results, ignore_index=True)
final_location_df.to_csv("Location_Features_Dataset4.csv", index=False)

print("✅ Location Feature Dataset Created Successfully!")
