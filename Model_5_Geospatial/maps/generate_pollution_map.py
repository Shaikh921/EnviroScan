import os
import pandas as pd
import folium

# Import layer modules
from hotspot_layer import add_hotspots  
from heatmap_layer import add_heatmap
from marker_layer import add_source_markers
from risk_layer import add_risk_zones


# -------------------------------
# Project Paths
# -------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "Dataset", "Final_Predictions.csv")

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "Model_5_Geospatial",
    "html_exports",
    "pollution_map.html"
)


# -------------------------------
# Load Dataset
# -------------------------------

print("Loading prediction dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print("Total rows:", len(df))


# -------------------------------
# Create Base Map
# -------------------------------

india_center = [20.5937, 78.9629]

pollution_map = folium.Map(
    location=india_center,
    zoom_start=5,
    tiles="CartoDB positron"
)

print("Base map created")


# -------------------------------
# Add Map Layers
# -------------------------------

print("Adding heatmap layer...")
add_heatmap(pollution_map, df)

print("Adding source markers...")
add_source_markers(pollution_map, df)

print("Adding high risk zones...")
add_risk_zones(pollution_map, df)

# print("Adding Hostspot layer...")
# add_hotspots(pollution_map, df)


# -------------------------------
# Layer Control
# -------------------------------

folium.LayerControl().add_to(pollution_map)

# -------------------------------
# Legend
# -------------------------------

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


# -------------------------------
# Save Map
# -------------------------------

pollution_map.save(OUTPUT_PATH)

print("Map exported successfully")
print("Location:", OUTPUT_PATH)