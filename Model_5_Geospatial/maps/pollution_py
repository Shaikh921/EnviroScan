import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster

print("Loading dataset...")

# ------------------------------------------------
# Load Dataset
# ------------------------------------------------
df = pd.read_csv("Final_Labeled_Pollution_Dataset.csv")

print("Dataset loaded")

# ------------------------------------------------
# Create Base Map
# ------------------------------------------------
center_lat = df["latitude"].mean()
center_lon = df["longitude"].mean()

pollution_map = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=6,
    tiles="OpenStreetMap"
)

print("Base map created")

# ------------------------------------------------
# Heatmap Layer (PM2.5 Intensity)
# ------------------------------------------------
heat_data = [[row["latitude"], row["longitude"], row["pm25"]] for index, row in df.iterrows()]

heatmap_layer = folium.FeatureGroup(name="PM2.5 Heatmap")

HeatMap(
    heat_data,
    radius=12,
    blur=15,
    max_zoom=10
).add_to(heatmap_layer)

heatmap_layer.add_to(pollution_map)

print("Heatmap added")

# ------------------------------------------------
# Marker Cluster for Pollution Sources
# ------------------------------------------------
marker_cluster = MarkerCluster(name="Pollution Sources")

# Color mapping
source_colors = {
    "Industrial": "red",
    "Vehicular": "blue",
    "Agricultural": "green",
    "Burning": "orange",
    "Natural": "purple"
}

for index, row in df.iterrows():

    source = row["pollution_source"]
    color = source_colors.get(source, "gray")

    popup_text = f"""
    <b>City:</b> {row['city']}<br>
    <b>Source:</b> {source}<br>
    <b>PM2.5:</b> {row['pm25']}<br>
    <b>Date:</b> {row['datetimeUtc']}
    """

    folium.Marker(
        location=[row["latitude"], row["longitude"]],
        popup=popup_text,
        icon=folium.Icon(color=color)
    ).add_to(marker_cluster)

marker_cluster.add_to(pollution_map)

print("Source markers added")

# ------------------------------------------------
# High Risk Zones
# ------------------------------------------------
high_risk_layer = folium.FeatureGroup(name="High Risk Zones")

high_risk = df[df["pm25"] > 150]

for index, row in high_risk.iterrows():

    folium.Circle(
        location=[row["latitude"], row["longitude"]],
        radius=3000,
        color="red",
        fill=True,
        fill_color="red",
        fill_opacity=0.4
    ).add_to(high_risk_layer)
high_risk_layer.add_to(pollution_map)

print("High risk zones added")

# ------------------------------------------------
# Layer Control
# ------------------------------------------------
folium.LayerControl().add_to(pollution_map)

# ------------------------------------------------
# Legend
# ------------------------------------------------
legend_html = '''
<div style="
position: fixed;
bottom: 50px;
left: 50px;
width: 200px;
height: 170px;
background-color: white;
border:2px solid grey;
z-index:9999;
font-size:14px;
padding:10px;
">

<b>Pollution Sources</b><br>
<i style="color:red">●</i> Industrial<br>
<i style="color:blue">●</i> Vehicular<br>
<i style="color:green">●</i> Agricultural<br>
<i style="color:orange">●</i> Burning<br>
<i style="color:purple">●</i> Natural

</div>
'''

pollution_map.get_root().html.add_child(folium.Element(legend_html))

# ------------------------------------------------
# Save Map
# ------------------------------------------------
pollution_map.save("pollution_map.html")

print("Map saved as pollution_map.html")