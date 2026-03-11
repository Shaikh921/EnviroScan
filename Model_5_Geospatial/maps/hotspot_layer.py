import folium


def add_hotspots(map_object, df):

    # Create a new layer
    hotspot_layer = folium.FeatureGroup(name="Pollution Hotspots")

    # Get top 10 highest PM2.5 locations
    top_hotspots = df.sort_values(by="pm25", ascending=False).head(10)

    for _, row in top_hotspots.iterrows():

        lat = row["latitude"]
        lon = row["longitude"]
        pm25 = row["pm25"]
        source = row["predicted_source"]
        city = row["city"]

        popup_text = f"""
        <b>🔥 Pollution Hotspot</b><br>
        <b>City:</b> {city}<br>
        <b>PM2.5:</b> {pm25}<br>
        <b>Source:</b> {source}
        """

        folium.Marker(
            location=[lat, lon],
            popup=popup_text,
            icon=folium.Icon(color="darkred", icon="warning-sign")
        ).add_to(hotspot_layer)

    hotspot_layer.add_to(map_object)