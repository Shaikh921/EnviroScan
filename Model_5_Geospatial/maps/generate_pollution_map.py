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
FALLBACK_DATA_PATH = os.path.join(PROJECT_ROOT, "Dataset", "Final_Dataset_Labeled_Balanced.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "Models", "XGBoost.joblib")
ENCODER_PATH = os.path.join(PROJECT_ROOT, "Models", "LabelEncoder.joblib")

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

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
elif os.path.exists(FALLBACK_DATA_PATH):
    print("Final_Predictions.csv not found. Loading fallback balanced dataset & predicting sources...")
    df = pd.read_csv(FALLBACK_DATA_PATH)
    features = ['co', 'no2', 'o3', 'pm10', 'pm25', 'so2', 'Temperature', 'Humidity', 'Wind Speed', 'Wind Direction', 'dist_to_road', 'dist_to_industry', 'dist_to_dump', 'dist_to_farmland']
    df = df.dropna(subset=features).reset_index(drop=True)
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        import joblib
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        preds = model.predict(df[features])
        df['predicted_source'] = encoder.inverse_transform(preds)
    elif 'pollution_source' in df.columns:
        df['predicted_source'] = df['pollution_source']
    else:
        df['predicted_source'] = 'Natural'
    # Save predictions file for future map runs
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved generated predictions to {DATA_PATH}")
else:
    raise FileNotFoundError("Neither Final_Predictions.csv nor Final_Dataset_Labeled_Balanced.csv found!")

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