# ==============================
# EnviroScan - XGBoost Model
# ==============================

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
df = pd.read_csv("Dataset/Final_Dataset_Labeled.csv")

# Feature Columns
features = ['co','no2','o3','pm10','pm25','so2',
            'Temperature','Humidity','Wind Speed','Wind Direction',
            'dist_to_road','dist_to_industry','dist_to_dump','dist_to_farmland']

X = df[features]
y = df['pollution_source']

# Encode Target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.4,
    random_state=42,
    stratify=y_encoded
)

# Create XGBoost Model
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5,
    reg_alpha=2,
    random_state=42,
    eval_metric='mlogloss'
)

# Train
xgb.fit(X_train, y_train)

# Predict
y_pred = xgb.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
