# EnviroScan Technical Documentation: AI-Based Pollution Source Identification System using Geospatial Analytics

---

## 1. Project Overview

**EnviroScan** is an artificial intelligence, data science, and geospatial analytics system designed to identify and classify sources of atmospheric pollution across major Indian cities. Traditional air quality monitoring stations measure ambient concentrations of pollutants such as $\text{PM}_{2.5}$, $\text{PM}_{10}$, $\text{NO}_2$, $\text{CO}$, $\text{SO}_2$, and $\text{O}_3$, but they do not dynamically pinpoint the specific emission activities causing deterioration in air quality.

EnviroScan addresses this bottleneck by integrating **pollution concentration data**, **meteorological parameters** (temperature, humidity, wind speed, wind direction), and **geospatial proximity features** (distances to primary/secondary roads, industrial zones, waste dumps, and agricultural land). Machine Learning models classify pollution sources into five distinct operational categories:
1. **Agricultural** (stubble burning, soil dust, fertilizer emissions)
2. **Burning** (open waste incineration, biomass combustion)
3. **Industrial** (smelter emissions, power plants, manufacturing exhaust)
4. **Natural** (dust storms, ambient baseline particulate matter)
5. **Vehicular** (internal combustion engine exhaust, brake/tire wear)

The system visualizes pollution trends, spatial hotspots, and model predictions through an interactive **Streamlit** web application integrated with **Folium** maps and **Plotly** visualizations.

---

## 2. Internship Context

* **Project Title**: EnviroScan — AI-Based Pollution Source Identification System using Geospatial Analytics
* **Domain**: Artificial Intelligence, Machine Learning, Geospatial Analytics, Environmental Data Science
* **Organization**: Infosys Springboard Internship Program
* **Core Objective**: Develop an end-to-end data processing, source classification, geospatial visualization, and web dashboard pipeline to provide actionable environmental intelligence.

---

## 3. Problem Statement

Air pollution monitoring faces three fundamental technical and practical challenges:

1. **Lack of Ground Truth Source Labels**: Public air quality monitoring networks (such as OpenAQ, CPCB, DPCC) record ambient concentrations but do not track source attribution. Standard supervised learning datasets for pollution source identification are non-existent.
2. **Overlapping Emission Signatures**: Industrial stacks, vehicular traffic, and waste burning often co-occur in urban dense areas, blending chemical profiles.
3. **Static & Fragmented Visualization**: Raw pollution tables fail to provide real-time spatial awareness or interactive risk assessments for local decision-makers.

---

## 4. Project Objectives

- Construct a unified data collection pipeline integrating OpenAQ air quality measurements, Open-Meteo meteorological parameters, and OpenStreetMap (OSMnx) geospatial infrastructure distances.
- Establish a rule-based source labeling engine based on chemical thresholds and spatial proximity rules.
- Train, tune, and evaluate ensemble machine learning models (Random Forest, Decision Tree, XGBoost) to classify pollution sources.
- Develop geospatial map overlays containing PM2.5 heatmaps, high-risk circle overlays, and interactive source marker clusters.
- Build an interactive, production-ready Streamlit dashboard providing real-time API sync, multi-city comparisons, historical trends, correlation matrix heatmaps, and forecasting.

---

## 5. System Architecture & Data Flow

### Conceptual Architecture

```text
       +--------------------+      +--------------------+      +--------------------+
       |  OpenAQ API / CSV  |      | Open-Meteo Weather |      | OSMnx / Overpass   |
       | (Pollutant Levels) |      | (Temp, Wind, Hum)  |      | (Roads, Industries)|
       +---------+----------+      +---------+----------+      +---------+----------+
                 |                           |                           |
                 +-------------------+-------+---------------------------+
                                     |
                                     v
                        +--------------------------+
                        | Data Collection Pipeline |
                        | (pollution_collection.py)|
                        +------------+-------------+
                                     |
                                     v
                        +--------------------------+
                        |     Data Preprocessing   |
                        | (Dataset_Cleaning.py)    |
                        +------------+-------------+
                                     |
                                     v
                        +--------------------------+
                        | Source Labeling Engine   |
                        | (Data_Lableing.py)       |
                        +------------+-------------+
                                     |
                                     v
                        +--------------------------+
                        | Model Training & Eval    |
                        | (EnviroScan_Model.py)    |
                        +------------+-------------+
                                     |
                                     v
                        +--------------------------+
                        | Saved Models (.joblib)   |
                        | (RandomForest, XGBoost)  |
                        +------------+-------------+
                                     |
                                     v
                        +--------------------------+
                        |  Streamlit Dashboard     |
                        | (Dashboard_Enhanced.py)  |
                        +------------+-------------+
                                     |
                     +---------------+---------------+
                     |                               |
                     v                               v
         +-----------------------+       +-----------------------+
         | Interactive Maps      |       | Predictions & Analytics|
         | (generate_map.py)     |       | (Plotly, AQI Gauges)  |
         +-----------------------+       +-----------------------+
```

### Actual Data Flow vs Documented Flow

| Step | Documented Workflow | Actual Code Implementation | Status / Mismatch |
| :--- | :--- | :--- | :--- |
| **1. Collection** | Automated API ingestion of OpenAQ + Weather + OSM | Individual scripts (`pollution_collection.py`, `weather_pollution_collection.py`, `location_collection.py`) exist; automated updater `auto_data_updater.py` created for OpenWeather API. | **Partial Match** |
| **2. Cleaning** | Outlier removal & city-level linear interpolation | `Dataset_Cleaning.py` contains commented-out outlier and interpolation logic; script saves to `Final_Dataset_Cleaned1.csv` instead of `Final_Dataset_Cleaned.csv`. | **Discrepancy** |
| **3. Labeling** | Rule-based assignment using pollutant percentiles | `Data_Lableing.py` uses 65th percentile thresholding and distance heuristics. | **Full Match** |
| **4. Training** | Train-test split (80:20) on clean balanced dataset | `EnviroScan_Model.py` adds 15% Gaussian noise, performs GridSearch/RandomSearch, saves models to `Models/`. | **Full Match** |
| **5. Geospatial** | Folium map with heatmaps & hotspot layers | `generate_pollution_map.py` requires `Dataset/Final_Predictions.csv` which is missing from repository; running script fails. | **Critical Failure** |
| **6. Dashboard** | `Model_6_Dashboard/Dashboard.py` | Dashboard implemented as `Model_6_Dashboard/Dashboard_Enhanced.py`. | **Filename Mismatch** |

---

## 6. Project Structure

```text
Environ_Scan_Project/
│
├── .env                              # Active environment variables (API Keys)
├── .env.example                      # Template environment variable file
├── .gitignore                        # Git exclusion rules
├── config.py                         # Centralized configuration, paths, & thresholds
├── HeatMap.py                        # Standalone exploratory data analysis script
├── LICENSE                           # MIT License file
├── packages.txt                      # System-level C/C++ geospatial libraries for deployment
├── Readme.md                         # Project README documentation
├── requirements.txt                  # Python dependencies specification
├── utils.py                          # Utility helper functions (API fetching, distance, cleaning)
│
├── Dataset/                          # Data storage directory
│   ├── Final_Dataset_Labeled_Balanced.csv  # Primary dataset (13,811 rows, 21 columns)
│   ├── city_pollution/               # 25 City CSV files (5 near-empty)
│   │   ├── Ahmedabad.csv
│   │   ├── Amritsar.csv
│   │   ├── Bengaluru.csv
│   │   ├── Chandigarh.csv
│   │   ├── Chennai.csv
│   │   ├── Delhi.csv
│   │   ├── Eloor.csv
│   │   ├── Fort_William.csv          # (371 bytes - header only)
│   │   ├── Gurugram.csv
│   │   ├── Hebbal_1st_Stage.csv      # (375 bytes - header only)
│   │   ├── Hyderabad.csv
│   │   ├── Jalna.csv
│   │   ├── Jawahar_Nagar.csv         # (363 bytes - header only)
│   │   ├── Karni_Colony.csv          # (355 bytes - header only)
│   │   ├── Kolkata.csv
│   │   ├── Madurai.csv
│   │   ├── Mumbai.csv
│   │   ├── Mysuru.csv
│   │   ├── Nagpur.csv
│   │   ├── Nehru_Nagar.csv           # (355 bytes - header only)
│   │   ├── Noida.csv
│   │   ├── Puducherry.csv
│   │   ├── Pune.csv
│   │   ├── Srinagar.csv
│   │   └── Vijayawada.csv
│   └── Script/                       # Data processing scripts
│       ├── auto_data_updater.py      # Real-time OpenWeather data sync script
│       ├── Data_Lableing.py          # Rule-based source labeling engine
│       ├── Dataset_Cleaning.py       # Data cleaning script (has commented out logic)
│       ├── final_dataset.py          # Dataset merging script
│       ├── location_collection.py    # OSMnx distance calculation script
│       ├── pollution_collection.py   # OpenAQ raw CSV aggregator
│       └── weather_pollution_collection.py # Open-Meteo historical weather fetcher
│
├── Images/                           # Image assets
│   ├── Matrix-XGBoost.png            # XGBoost confusion matrix image
│   └── Random_forest.png             # Random Forest confusion matrix image
│
├── ModelScript/                      # Machine Learning pipeline
│   ├── DT_Confusion_Matrix.png       # Decision tree confusion matrix plot
│   ├── EnviroScan_Model.py           # Main integrated model training script
│   ├── Features_Importance.png       # Feature importance bar plot
│   ├── RF_Confusion_Matrix.png       # Random Forest confusion matrix plot
│   ├── XGBOOST_Confusion_Matrix.png  # XGBoost confusion matrix plot
│   └── UniqueScript/                 # Modular step-by-step experiment scripts
│       ├── decision_tree.py
│       ├── features_importance.py
│       ├── hyperparameter_tuning.py
│       ├── preprocessing_and_split.py
│       ├── random_forest.py
│       ├── save_model.py
│       └── Xgboost.py
│
├── Model_5_Geospatial/               # Geospatial mapping module
│   ├── html_exports/
│   │   └── pollution_map.html        # Exported standalone Folium HTML map (32.8 MB)
│   └── maps/
│       ├── generate_pollution_map.py # Main map generator script (fails: missing CSV)
│       ├── heatmap_layer.py          # Folium HeatMap plugin wrapper
│       ├── hotspot_layer.py          # Top 10 PM2.5 marker overlay
│       ├── marker_layer.py           # MarkerCluster plugin wrapper
│       ├── pollution_py              # Standalone map script (missing .py extension)
│       └── risk_layer.py             # LinearColormap Circle risk layer wrapper
│
├── Model_6_Dashboard/                # Web Dashboard application
│   └── Dashboard_Enhanced.py         # Advanced Streamlit application (1392 lines)
│
└── Models/                           # Serialized Joblib model files
    ├── DecisionTree.joblib           # Trained DecisionTree model (20.9 KB)
    ├── LabelEncoder.joblib           # Fitted LabelEncoder (533 bytes)
    ├── RandomForest.joblib           # Trained RandomForest model (2.37 MB)
    └── XGBoost.joblib                # Trained XGBoost model (709.7 KB)
```

---

## 7. Technologies and Frameworks Analysis

| Technology | Version / Range | Purpose in EnviroScan | Where Used | Key Functions / Classes | Interaction with Components |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Python** | `3.13.5` | Core programming language | Entire codebase | `pathlib`, `logging`, `sys`, `json`, `io` | Foundation for execution |
| **Pandas** | `>=2.0.0, <2.3.0` | Data manipulation, tabular ingestion, merging | All `.py` scripts | `read_csv()`, `merge()`, `pivot_table()`, `quantile()`, `isna()` | Data pipeline backbone |
| **NumPy** | `>=1.24.0, <2.0.0` | Vectorized math, array operations, distance metrics | `utils.py`, `config.py`, ML scripts | `radians()`, `random.normal()`, `arange()` | Array computations for BallTree and noise |
| **Scikit-Learn**| `>=1.3.0, <1.6.0` | Machine learning algorithms, evaluation, splitting | `ModelScript/`, `utils.py` | `RandomForestClassifier`, `DecisionTreeClassifier`, `train_test_split`, `GridSearchCV`, `LabelEncoder`, `BallTree`, `confusion_matrix`, `classification_report` | Core ML modeling & distance spatial trees |
| **XGBoost** | `>=2.0.0, <2.2.0` | Gradient boosted decision trees classifier | `ModelScript/EnviroScan_Model.py`, `Dashboard_Enhanced.py` | `XGBClassifier(eval_metric='mlogloss')` | Primary production classifier for Dashboard predictions |
| **Streamlit** | `>=1.28.0` | Web Application Dashboard framework | `Model_6_Dashboard/Dashboard_Enhanced.py` | `set_page_config()`, `cache_data`, `cache_resource`, `sidebar`, `columns()`, `metric()`, `plotly_chart()`, `dataframe()`, `download_button()` | User interface, controls, and presentation |
| **Folium** | `>=0.14.0` | Interactive map generation | `Model_5_Geospatial/`, `Dashboard_Enhanced.py` | `Map()`, `FeatureGroup`, `Marker()`, `Circle()`, `LayerControl()` | Geospatial rendering |
| **streamlit-folium**| `>=0.15.0` | Streamlit integration for Folium maps | `Dashboard_Enhanced.py` | `st_folium()` | Rendering interactive maps inside Streamlit tabs |
| **Plotly** | `>=5.14.0` | Interactive charting engine | `Dashboard_Enhanced.py` | `px.pie()`, `px.bar()`, `go.Figure()`, `go.Indicator()`, `go.Heatmap()`, `go.Scatter()` | Gauges, time-series, correlation matrix heatmaps |
| **Matplotlib** | `>=3.7.0, <3.10.0` | Static plot generation for confusion matrices | `ModelScript/`, `HeatMap.py` | `plt.figure()`, `plt.savefig()`, `plt.show()` | Saving confusion matrix and feature importance PNGs |
| **Seaborn** | `>=0.12.0` | Statistical heatmaps | `ModelScript/`, `HeatMap.py` | `sns.heatmap()`, `sns.histplot()`, `sns.boxplot()` | Matrix visualization during model evaluation |
| **Joblib** | `>=1.3.0` | Model serialization and deserialization | `ModelScript/`, `Dashboard_Enhanced.py` | `joblib.dump()`, `joblib.load()` | Saving & loading `.joblib` model binaries |
| **OSMnx** | Not pinned in `requirements.txt` | OpenStreetMap geospatial feature extraction | `Dataset/Script/location_collection.py` | `ox.features_from_point()` | Querying road, industrial, dump, and farmland spatial centroids |

---

## 8. Data Sources & Ingestion

1. **OpenAQ API / CSV Exports**: Provides concentration values for $\text{PM}_{2.5}$, $\text{PM}_{10}$, $\text{NO}_2$, $\text{CO}$, $\text{SO}_2$, $\text{O}_3$ collected from monitoring stations across 25 Indian cities.
2. **Open-Meteo Historical Weather API**: Ingests hourly historical weather metrics ($\text{temperature\_2m}$, $\text{relative\_humidity\_2m}$, $\text{wind\_speed\_10m}$, $\text{wind\_direction\_10m}$) matching the coordinates and date timestamps of pollution stations.
3. **OpenStreetMap via OSMnx**: Queries point-of-interest geometries within a 20 km radius around city centroids for:
   - Roads (`highway=["primary", "secondary"]`)
   - Industrial land (`landuse="industrial"`)
   - Waste dumps (`amenity="landfill"`)
   - Farmlands (`landuse="farmland"`)

---

## 9. Data Preprocessing Pipeline

1. **Station Aggregation**: In `pollution_collection.py`, raw station data is filtered for required pollutants and pivoted into wide format with columns (`city`, `latitude`, `longitude`, `co`, `no2`, `o3`, `pm10`, `pm25`, `so2`, `datetimeUtc`).
2. **Spatial Distance Calculation**: In `location_collection.py` and `utils.py`, Haversine distances from each station coordinate to the nearest road, industrial zone, landfill, and farmland centroid are computed using `sklearn.neighbors.BallTree`:
   $$\text{distance} = R \cdot \text{haversine\_query}(\text{points}, \text{features})$$
   where $R = 6,371,000 \text{ meters}$.
3. **Imputation & Outlier Handling**: Missing values in pollutant columns are interpolated using linear temporal interpolation followed by forward/backward filling (`ffill()`, `bfill()`).

---

## 10. Feature Engineering

The dataset comprises 14 core numerical features for machine learning:

$$\mathbf{X} = [ \underbrace{\text{co}, \text{no2}, \text{o3}, \text{pm10}, \text{pm25}, \text{so2}}_{\text{6 Pollutant Features}}, \quad \underbrace{\text{Temperature}, \text{Humidity}, \text{Wind Speed}, \text{Wind Direction}}_{\text{4 Weather Features}}, \quad \underbrace{\text{dist\_to\_road}, \text{dist\_to\_industry}, \text{dist\_to\_dump}, \text{dist\_to\_farmland}}_{\text{4 Geospatial Proximity Features}} ]$$

In `EnviroScan_Model.py`, controlled Gaussian noise is added to feature columns during training to prevent exact lookup memorization:
$$x_{\text{noisy}} = x + \mathcal{N}(0, \sigma_{\text{noise}} \cdot \sigma_x), \quad \sigma_{\text{noise}} = 0.15$$

---

## 11. Source Labeling Methodology (Rule-Based Engine)

Because real-world ground truth source labels were absent, `Dataset/Script/Data_Lableing.py` computes the 65th percentile threshold ($\text{P}_{65}$) across all pollutant distributions:

$$\text{so2}_{\text{high}} = \text{Quantile}_{0.65}(\text{SO}_2), \quad \text{no2}_{\text{high}} = \text{Quantile}_{0.65}(\text{NO}_2), \quad \text{pm25}_{\text{high}} = \text{Quantile}_{0.65}(\text{PM}_{2.5})$$
$$\text{co}_{\text{high}} = \text{Quantile}_{0.65}(\text{CO}), \quad \text{pm10}_{\text{high}} = \text{Quantile}_{0.65}(\text{PM}_{10})$$

### Rule Evaluation Cascade

```text
IF so2 > so2_high AND dist_to_industry < 2000m:
    RETURN "Industrial"
ELSE IF no2 > no2_high AND dist_to_road < 600m:
    RETURN "Vehicular"
ELSE IF pm25 > pm25_high AND co > co_high AND dist_to_dump < 6000m:
    RETURN "Burning"
ELSE IF pm10 > pm10_high AND dist_to_farmland < 4000m AND Humidity < 70%:
    RETURN "Agricultural"
ELSE:
    RETURN "Natural"
```

> [!WARNING]
> **Artificially High Performance Impact**: Because target labels $\mathbf{y}$ are derived directly from deterministic threshold functions of $\mathbf{X}$, decision tree models learn these decision boundaries with near 98-99% accuracy. This performance reflects rule-boundary reconstruction rather than real-world physical generalization.

---

## 12. Machine Learning Pipeline & Model Development

### Model Comparison & Verification Results

Models were re-evaluated on `Dataset/Final_Dataset_Labeled_Balanced.csv` (13,801 valid rows after dropping 6 missing distance rows):

| Model Name | Train Accuracy | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Overfitting Delta ($\text{Train} - \text{Test}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.9721 | 0.9678 | 0.9681 | 0.9678 | 0.9649 | +0.0043 |
| **XGBoost Classifier** | **0.9784** | **0.9797** | **0.9797** | **0.9797** | **0.9792** | **-0.0013** |
| **Decision Tree** | 0.9741 | 0.9750 | 0.9755 | 0.9750 | 0.9748 | -0.0009 |

* **Best Model**: **XGBoost Classifier** (highest test accuracy 97.97% and F1-Score 0.9792).

### Class Imbalance Breakdown in "Balanced" Dataset

| Source Class | Record Count | Percentage | Precision (XGB) | Recall (XGB) | F1-Score (XGB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Natural** | 7,619 | 55.21% | 0.98 | 1.00 | 0.99 |
| **Industrial** | 3,100 | 22.46% | 0.99 | 0.97 | 0.98 |
| **Vehicular** | 2,497 | 18.09% | 0.97 | 0.98 | 0.98 |
| **Burning** | 321 | 2.33% | 0.92 | 0.92 | 0.92 |
| **Agricultural** | 264 | 1.91% | 0.95 | 0.70 | 0.80 |

---

## 13. Streamlit Dashboard Architecture

The dashboard (`Model_6_Dashboard/Dashboard_Enhanced.py`) is structured into four primary interactive view modes and four analytic tabs:

### View Modes
1. **Single City**: Displays current pollutant metrics, AQI gauge chart (0-300 scale), real-time weather metrics, time-series line trends, and real-time model prediction.
2. **Multi-City Comparison**: Enables multi-select comparison (2 to 5 cities) across PM2.5, PM10, NO2, CO, SO2, O3 with grouped bar charts and ranking tables.
3. **Historical Analysis**: Time-series filtering (Last 7 Days, 30 Days, 90 Days, Custom Range), moving averages, histogram distributions, box plots, and a $6 \times 6$ interactive Plotly correlation matrix heatmap.
4. **Pollution Forecast**: Linear regression forecasting for 6-48 hours into the future for PM2.5 with automated health recommendation alerts.

### Analytic Tabs
- **Tab 1: Pollution Map**: Embedded HTML Folium map (`pollution_map.html`).
- **Tab 2: Source Distribution**: Interactive pie chart and frequency bar chart of predicted pollution sources.
- **Tab 3: AI Model Analysis**: Visual displays of confusion matrices and feature importance breakdowns.
- **Tab 4: Download Report**: Instant export of filtered data to CSV or Excel (`.xlsx`).

---

## 14. Code Walkthrough (Important Snippets)

### Snippet 1: Spatial BallTree Haversine Distance (`utils.py`)

```python
def compute_distance(points_df: pd.DataFrame, 
                    feature_coords: np.ndarray, 
                    column_name: str) -> pd.DataFrame:
    """Compute haversine distance from points to nearest features in meters"""
    if feature_coords.shape[0] == 0:
        points_df[column_name] = np.nan
        return points_df
    
    points_rad = np.radians(points_df[['latitude', 'longitude']].values)
    features_rad = np.radians(feature_coords)
    
    tree = BallTree(features_rad, metric='haversine')
    dist, _ = tree.query(points_rad, k=1)
    
    # Earth radius = 6,371,000 meters
    points_df[column_name] = dist.flatten() * 6371000
    return points_df
```
* **Explanation**: Uses Scikit-Learn's `BallTree` with spherical haversine metric to query the closest spatial feature (e.g. road or factory) for thousands of coordinates in $O(N \log M)$ time.

---

### Snippet 2: Model Prediction in Dashboard (`Dashboard_Enhanced.py`)

```python
# Prepare feature input DataFrame matching exact model column order
input_data = pd.DataFrame([{
    "co": co,
    "no2": no2,
    "o3": o3,
    "pm10": pm10,
    "pm25": pm25,
    "so2": so2,
    "Temperature": weather["Temperature"],
    "Humidity": weather["Humidity"],
    "Wind Speed": weather["Wind Speed"],
    "Wind Direction": weather["Wind Direction"],
    "dist_to_road": city_row["dist_to_road"],
    "dist_to_industry": city_row["dist_to_industry"],
    "dist_to_dump": city_row["dist_to_dump"],
    "dist_to_farmland": city_row["dist_to_farmland"]
}])

# Model inference & inverse label encoding
prediction = model.predict(input_data)
source = encoder.inverse_transform(prediction)
st.success(f"**{source[0]}**")
```
* **Explanation**: Constructs a single-row DataFrame with all 14 features in exact order expected by `XGBoost.joblib`, runs `model.predict()`, and maps the integer output back to class text using `LabelEncoder`.

---

## 15. Comprehensive Test Report

| Test ID | Component | Test Case Description | Expected Result | Actual Result | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **TEST-01** | Environment | Python compilation check on all `.py` files | All 24 files compile cleanly | 24 python files compiled cleanly | **PASS** |
| **TEST-02** | Config | Configuration validation (`validate_config()`) | Returns True when paths & API key set | Config validates successfully | **PASS** |
| **TEST-03** | Model Loading | Load `RandomForest.joblib`, `XGBoost.joblib`, `DecisionTree.joblib`, `LabelEncoder.joblib` | All models load with 14 input features | All 4 joblib binaries loaded successfully; `n_features_in_ == 14` | **PASS** |
| **TEST-04** | Dataset | Verify `Dataset/Final_Dataset_Labeled_Balanced.csv` existence & schema | File exists with 21 columns | File exists (13,811 rows, 21 columns) | **PASS** |
| **TEST-05** | Dataset | Verify existence of intermediate datasets (`Final_Dataset_Cleaned.csv`, `Final_Predictions.csv`, etc.) | Files exist in `Dataset/` folder | Files are missing from `Dataset/` directory | <span style="color:red; font-weight:bold;">FAIL</span> |
| **TEST-06** | City Data | Inspect city CSV files in `Dataset/city_pollution/` | All 25 CSV files contain full pollution records | 5 files (`Fort_William.csv`, `Hebbal_1st_Stage.csv`, etc.) contain <380 bytes (header only) | **WARNING** |
| **TEST-07** | Map Generator | Execute `Model_5_Geospatial/maps/generate_pollution_map.py` | Generates `pollution_map.html` | Crashes with `FileNotFoundError` (missing `Final_Predictions.csv`) | <span style="color:red; font-weight:bold;">FAIL</span> |
| **TEST-08** | File Extension | Verify script extension in `Model_5_Geospatial/maps/pollution_py` | Extension should be `.py` | File is named `pollution_py` without extension | <span style="color:red; font-weight:bold;">FAIL</span> |
| **TEST-09** | Script Typo | Check column renaming in `pollution_collection.py` | Renames `datetimeUtc` to `timestamp` | Typo `datetimUtc` causes rename to fail silently | <span style="color:red; font-weight:bold;">FAIL</span> |
| **TEST-10** | Script Output | Check save path in `Dataset_Cleaning.py` | Saves cleaned CSV to `Final_Dataset_Cleaned.csv` | Saves to `Final_Dataset_Cleaned1.csv` while printing `Final_Dataset_Cleaned.csv` | <span style="color:red; font-weight:bold;">FAIL</span> |
| **TEST-11** | Model Performance| Evaluate XGBoost on test dataset | Test accuracy > 90% | Test Accuracy: 97.97%, F1-Score: 0.9792 | **PASS** |
| **TEST-12** | Model Performance| Evaluate Random Forest on test dataset | Test accuracy > 90% | Test Accuracy: 96.78%, F1-Score: 0.9649 | **PASS** |
| **TEST-13** | Dashboard | Load `Dashboard_Enhanced.py` dependencies | All Streamlit, Folium, Plotly packages import | Imports succeeded cleanly | **PASS** |
| **TEST-14** | Data Updater | Execute `auto_data_updater.py --mode once` | Syncs OpenWeather API data | Functioning with valid API key | **PASS** |
| **TEST-15** | Security | Check repository for committed API keys | Secrets managed via environment variables | `OPENWEATHER_KEY` hardcoded in `.env` file | **WARNING** |

---

## 16. Bugs and Issues Found

### Issue 1: Missing Required Dataset Files
- **Evidence**: `Dataset/Final_Predictions.csv`, `Dataset/Final_Dataset_Cleaned.csv`, `Dataset/Final_Dataset_Labeled.csv` do not exist on disk.
- **Impact**: Map generation script `generate_pollution_map.py` crashes on startup.
- **Recommended Fix**: Update `generate_pollution_map.py` to fallback to `Final_Dataset_Labeled_Balanced.csv` and generate missing predicted source column if `Final_Predictions.csv` is absent.

### Issue 2: Filename Typo in Geospatial Map Script
- **Evidence**: File `Model_5_Geospatial/maps/pollution_py` is missing its `.py` extension.
- **Impact**: Cannot be imported or executed directly via Python standard tools.
- **Recommended Fix**: Rename `pollution_py` to `pollution.py`.

### Issue 3: Misspelled Column Key in Data Aggregator
- **Evidence**: In `pollution_collection.py` line 34: `df_pivot.rename(columns={'datetimUtc': 'timestamp'}, inplace=True)`.
- **Impact**: Column `datetimeUtc` is never renamed to `timestamp`.
- **Recommended Fix**: Correct string to `datetimeUtc`.

### Issue 4: Mismatch in Model Filename References
- **Evidence**: `save_model.py` outputs `Models/pollution_source_model.joblib`, whereas `config.py` and `EnviroScan_Model.py` write `Models/RandomForest.joblib`.
- **Impact**: Confusion and potential `FileNotFoundError` if running `save_model.py` directly.
- **Recommended Fix**: Standardize all scripts to use `config.py` path constants.

### Issue 5: Discrepancy Between README and Implementation
- **Evidence**: README points to `Model_6_Dashboard/Dashboard.py`, but actual dashboard file is `Model_6_Dashboard/Dashboard_Enhanced.py`.
- **Impact**: New users following README instructions receive command error.
- **Recommended Fix**: Update `Readme.md` command to `streamlit run Model_6_Dashboard/Dashboard_Enhanced.py`.

---

## 17. Recommended Improvement Roadmap

### Critical Priority (Immediate Fixes)
1. **Fix Map Generator Script**: Update `generate_pollution_map.py` to read `Final_Dataset_Labeled_Balanced.csv` and compute model predictions dynamically.
2. **Rename File**: Rename `Model_5_Geospatial/maps/pollution_py` to `pollution.py`.
3. **Fix Column Rename Typo**: Correct `datetimUtc` typo in `pollution_collection.py`.

### High Priority
1. **Handle Class Imbalance**: Implement SMOTE or class weighting (`scale_pos_weight` in XGBoost / `class_weight='balanced'` in RF) to boost recall for minority classes (Agricultural & Burning).
2. **Standardize Path Constants**: Enforce `config.py` usage across all scripts in `Dataset/Script/` and `ModelScript/UniqueScript/`.
3. **Clean Empty City Files**: Populate or filter out the 5 near-empty city CSV files.

### Medium Priority
1. **Update README.md**: Align execution instructions, script paths, and screenshot links with actual codebase structure.
2. **Add Unit Tests**: Add pytest suite in `tests/` directory covering data cleaning, distance calculation, and model loading.

### Low Priority
1. **Enhanced Forecasting**: Replace single-variable linear regression forecast in Dashboard with Prophet or SARIMAX time-series model.

---

## 18. Security Review

> [!WARNING]
> **Hardcoded API Credentials**: File `.env` contains an active OpenWeather API key (`OPENWEATHER_KEY=339...`). While `.env` is listed in `.gitignore`, committing any real credentials poses security risks if pushed to repository mirrors.
> **Recommended Action**: Replace `.env` content with placeholder text and ensure developers create `.env` locally from `.env.example`.

### Recommended `.gitignore` Entries
```text
.env
.env.local
.env.*.local
.streamlit/secrets.toml
__pycache__/
*.pyc
*.joblib
*.html
```

---

## 19. Deployment Guide

### Deployment Platforms
- **Streamlit Community Cloud**: Fully supported (via `requirements.txt` and `packages.txt`).
- **Local Deployment**: Supported on Windows / Linux / macOS.

### Steps for Deployment
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Set Environment Variables**:
   Create `.env` file containing:
   ```env
   OPENWEATHER_KEY=your_openweather_api_key_here
   ```
3. **Launch Streamlit Dashboard**:
   ```bash
   streamlit run Model_6_Dashboard/Dashboard_Enhanced.py
   ```

---

## 20. Limitations

1. **Rule-Based Ground Truth**: Target labels are derived from deterministic rules, artificially inflating model accuracy metrics (97-98%).
2. **Severe Class Imbalance**: Natural pollution source accounts for 55.2% of data, while Agricultural accounts for under 2%.
3. **Linear Regression Forecasting**: Current 24-hour PM2.5 forecast uses basic linear trend fitting rather than temporal seasonal models.

---

## 21. Future Enhancements

- Ingest satellite remote sensing data (Sentinel-5P TROPOMI $\text{NO}_2$ and $\text{SO}_2$ columns).
- Implement SMOTE oversampling and cost-sensitive learning for minority pollution classes.
- Upgrade forecasting module to LSTM / Prophet neural time-series forecasting.
- Deploy automated Docker container pipeline for continuous ingestion.

---

## 22. Conclusion

EnviroScan provides a comprehensive data science and geospatial analytical framework for identifying pollution sources. The application features a functional model pipeline (with XGBoost achieving 97.97% test accuracy) and an interactive Streamlit dashboard. By addressing the identified script typos, dataset path fallbacks, and class imbalance, EnviroScan can serve as a robust platform for real-time air quality monitoring and source attribution.
