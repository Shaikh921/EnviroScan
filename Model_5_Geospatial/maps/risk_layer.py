import folium
import branca.colormap as cm


def add_risk_zones(map_object, df):

    risk_layer = folium.FeatureGroup(name="High Risk Zones")

    # PM2.5 Color Gradient
    colormap = cm.LinearColormap(
        colors=["green", "yellow", "red"],
        vmin=df["pm25"].min(),
        vmax=df["pm25"].max(),
    )

    colormap.caption = "PM2.5 Pollution Level"

    # Aggregate by location to avoid rendering tens of thousands of overlapping circles
    loc_df = df.groupby(["latitude", "longitude", "city"], as_index=False)["pm25"].mean()

    for _, row in loc_df.iterrows():

        lat = row["latitude"]
        lon = row["longitude"]
        pm25 = row["pm25"]

        color = colormap(pm25)

        folium.Circle(
            location=[lat, lon],
            radius=5000,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=f"<b>City:</b> {row['city']}<br><b>Avg PM2.5:</b> {pm25:.1f}"
        ).add_to(risk_layer)

    risk_layer.add_to(map_object)

    # Add Color Scale to Map
    colormap.add_to(map_object)