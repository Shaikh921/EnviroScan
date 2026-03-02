# 🌍 EnviroScan — Pollution Source Classification Project

## 📌 Project Overview

EnviroScan is an end-to-end machine learning project that identifies pollution sources using environmental data collected from major Indian cities.

The project includes:

- Environmental data collection  
- Data cleaning and feature engineering  
- Rule-based labeling of pollution sources  
- Machine learning model training and evaluation  
- Model comparison (Decision Tree, Random Forest, XGBoost)  
- Saving trained models for future deployment  

The final objective is to classify pollution sources such as traffic, industrial activity, or dumping areas based on environmental indicators.

---

## 📊 Data Sources

### 1️⃣ Air Pollution Data  
- Source: OpenAQ API  
- Website: https://api.openaq.org/  
- Pollutants Collected:
  - PM2.5  
  - PM10  
  - NO₂  
  - CO  
  - SO₂  
  - O₃  

### 2️⃣ Weather Data  
- Source: OpenWeatherMap API  
- Website: https://openweathermap.org/api  
- Features Collected:
  - Temperature  
  - Humidity  
  - Wind Speed  
  - Pressure  

### 3️⃣ Location-Based Features  
- Source: OpenStreetMap (OSM)  
- Tool Used: OSMnx + Overpass API  
- Extracted Features:
  - Distance to roads  
  - Distance to industrial areas  
  - Distance to dumping sites  
  - Distance to farmland  
  - Land-use patterns  

---

## 🏙 Cities Covered

Pune, Mumbai, Delhi, Srinagar, Ahmedabad, Nagpur, Amritsar, Bengaluru, Chandigarh, Chennai, Eloor, Gurugram, Hyderabad, Jalna, Kolkata, Madurai, Mysuru, Noida, Puducherry, Vijayawada.

These cities represent diverse pollution patterns across India.

---

## ⏳ Time Range

- Data collected over 1 year.

---

## 🧹 Data Processing

### ✔ Data Cleaning
- Removed missing values  
- Removed unrealistic zero values  
- Standardized feature naming  

### ✔ Feature Engineering
- Merged pollution + weather datasets  
- Generated distance-based environmental features  
- Created final structured dataset  

### ✔ Rule-Based Labeling
Since real-world pollution source labels were unavailable, rule-based labeling was applied using pollutant thresholds and proximity features.

> ⚠ Limitation: Labels are synthetic and may not fully represent real-world conditions.

---

## 🤖 Machine Learning Pipeline

The training script performs:

1. Load labeled dataset  
2. Clean data  
3. Feature selection  
4. Label encoding  
5. Controlled noise injection  
6. Train-test split (80-20, stratified)  
7. Hyperparameter tuning using GridSearchCV  
8. Train multiple models  
9. Evaluate models  
10. Save trained models  

---

## 🧠 Models Trained

### 🌲 Random Forest
- Hyperparameter tuned
- Feature importance extracted

### 🌳 Decision Tree
- Hyperparameter tuned
- Used for baseline comparison

### 🚀 XGBoost
- Gradient boosting model
- Used for performance comparison

### 📈 Evaluation Metrics
- Accuracy  
- Weighted F1-score  
- Classification Report  
- Confusion Matrix  

---

## 💾 Model Saving

Models are saved in the `Models/` directory:

- `RandomForest_*.joblib`
- `DecisionTree_*.joblib`
- `XGBoost_*.joblib`
- `LabelEncoder_*.joblib`

These can be loaded later for prediction or deployment.

---

## 📁 Project Structure
ENVIRON_SCAN_PROJECT/
│
├── Dataset/
│ ├── city_pollution/
│ └── Final_Dataset_Labeled.csv
│
├── Models/
│ ├── RandomForest_.joblib
│ ├── DecisionTree_.joblib
│ ├── XGBoost_.joblib
│ └── LabelEncoder_.joblib
│
├── Script/
│ ├── data_collection/
│ ├── feature_engineering/
│ └── train_models.py
│
├── README.md
└── LICENSE

---

## ⚠ Current Limitations

- Rule-based synthetic labels  
- Possible optimistic accuracy  
- No real-time streaming integration   

---

## 🚀 Future Improvements

- Use real-world verified pollution labels  
- Advanced hyperparameter tuning  
- API deployment  
- Model monitoring system  
- Real-time prediction pipeline  

---

## 📌 Summary

EnviroScan evolved from a data collection project into a complete machine learning classification system.

It demonstrates:

- End-to-end ML workflow  
- Model comparison  
- Structured experimentation  
- Reproducible model saving  