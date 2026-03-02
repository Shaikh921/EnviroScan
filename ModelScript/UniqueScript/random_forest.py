# ==============================
# EnviroScan - Week 4
# Step 5: Random Forest Model
# ==============================

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------
# Load Dataset
# ------------------------------
df = pd.read_csv("Dataset/Final_Dataset_Labeled.csv")

X = df[['co','no2','o3','pm10','pm25','so2',
        'Temperature','Humidity','Wind Speed','Wind Direction',
        'dist_to_road','dist_to_industry','dist_to_dump','dist_to_farmland']]

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

# ------------------------------
# Train Random Forest
# ------------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

# ------------------------------
# Predictions
# ------------------------------
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)

# ------------------------------
# Evaluation
# ------------------------------
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=le.classes_))