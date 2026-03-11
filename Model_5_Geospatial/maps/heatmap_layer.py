import folium
from folium.plugins import HeatMap


def add_heatmap(map_object, df):

    # Create a feature group for the heatmap
    heatmap_layer = folium.FeatureGroup(name="Pollution Heatmap")

    # Prepare heatmap data (lat, lon, intensity)
    heat_data = []

    for _, row in df.iterrows():

        lat = row["latitude"]
        lon = row["longitude"]
        pm25 = row["pm25"]

        heat_data.append([lat, lon, pm25])

    # Create heatmap
    HeatMap(
        heat_data,
        radius=15,
        blur=20,
        max_zoom=10
    ).add_to(heatmap_layer)

    # Add layer to map
    heatmap_layer.add_to(map_object)