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

    for _, row in df.iterrows():

        lat = row["latitude"]
        lon = row["longitude"]
        pm25 = row["pm25"]

        color = colormap(pm25)

        folium.Circle(
            location=[lat, lon],
            radius=4000,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=f"PM2.5 Level: {pm25}"
        ).add_to(risk_layer)

    risk_layer.add_to(map_object)

    # Add Color Scale to Map
    colormap.add_to(map_object)