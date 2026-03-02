# ==============================
# EnviroScan - Week 4
# Step 8: Save Final Model
# ==============================

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("Dataset/Final_Dataset_Labeled.csv")

features = ['co','no2','o3','pm10','pm25','so2',
            'Temperature','Humidity','Wind Speed','Wind Direction',
            'dist_to_road','dist_to_industry','dist_to_dump','dist_to_farmland']

X = df[features]
y = df['pollution_source']

# Encode
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Train Final Model
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    random_state=42
)

rf.fit(X_train, y_train)

# Save Model
joblib.dump(rf, "Models/pollution_source_model.joblib")

# Save Label Encoder
joblib.dump(le, "Models/label_encoder.joblib")

print("Model and Encoder saved successfully ✅")