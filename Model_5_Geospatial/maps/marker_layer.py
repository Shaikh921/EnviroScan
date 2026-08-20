import folium
from folium.plugins import MarkerCluster


# -------------------------------
# Source Color Mapping
# -------------------------------

def get_source_color(source):

    source_colors = {
        "Industrial": "red",
        "Vehicular": "blue",
        "Agricultural": "green",
        "Burning": "orange",
        "Natural": "purple"
    }

    return source_colors.get(source, "gray")


# -------------------------------
# Add Source Markers with Cluster
# -------------------------------

def add_source_markers(map_object, df):

    marker_layer = folium.FeatureGroup(name="Pollution Sources")

    marker_cluster = MarkerCluster()

    # Deduplicate by city location taking most recent record per location
    if 'datetimeUtc' in df.columns:
        marker_df = df.sort_values('datetimeUtc').groupby(['city', 'latitude', 'longitude'], as_index=False).last()
    else:
        marker_df = df.groupby(['city', 'latitude', 'longitude'], as_index=False).first()

    for _, row in marker_df.iterrows():

        lat = row["latitude"]
        lon = row["longitude"]
        source = row.get("predicted_source", row.get("pollution_source", "Natural"))
        pm25 = row["pm25"]
        city = row["city"]
        time = row.get("datetimeUtc", "Latest")

        popup_text = f"""
        <b>City:</b> {city}<br>
        <b>Predicted Source:</b> {source}<br>
        <b>PM2.5:</b> {pm25:.1f}<br>
        <b>Date:</b> {time}
        """

        folium.Marker(
            location=[lat, lon],
            popup=popup_text,
            icon=folium.Icon(color=get_source_color(source))
        ).add_to(marker_cluster)

    marker_cluster.add_to(marker_layer)
    marker_layer.add_to(map_object)