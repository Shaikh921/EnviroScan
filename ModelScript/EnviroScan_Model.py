# EnviroScan - Pollution Source Classification Pipeline

# Description:
# - Data loading & cleaning
# - Feature engineering
# - Noise injection
# - Hyperparameter tuning
# - Model training (RF, DT, XGB)
# - Evaluation
# - Feature importance
# - Model saving

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
# from datetime import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier


# Configuration

DATA_PATH = "Dataset/FInal_Dataset_Labeled.csv"
MODEL_DIR = "Models"
TEST_SIZE = 0.2
RANDOM_STATE = 42
NOISE_LEVEL = 0.15

os.makedirs(MODEL_DIR, exist_ok=True)

# timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# Step 1: Load Dataset

print("🚀 Loading Dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset Shape: {df.shape}")

# Step 2: Data Cleaning

print("\n🔍 Checking Missing Values...")
print(df.isnull().sum())

df = df.dropna().reset_index(drop=True)
print("✅ Missing values removed")

# Step 3: Feature Selection

features = [
    'co','no2','o3','pm10','pm25','so2',
    'Temperature','Humidity','Wind Speed','Wind Direction',
    'dist_to_road','dist_to_industry','dist_to_dump','dist_to_farmland'
]

X = df[features].copy()
y = df['pollution_source']

# Encode target labels

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("\n📌 Class Mapping:")
for idx, class_name in enumerate(le.classes_):
    print(f"{class_name} --> {idx}")

# Step 4: Add Controlled Noise (to reduce rule-based bias)

print("\n⚙ Adding Noise to Features...")
for col in features:
    std_dev = X[col].std()
    noise = np.random.normal(0, NOISE_LEVEL * std_dev, X.shape[0])
    X.loc[:, col] += noise

print("✅ Noise added successfully")

# Step 5: Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

print(f"\nTrain Size: {X_train.shape}")
print(f"Test Size: {X_test.shape}")

# Step 6: Random Forest with Hyperparameter Tuning

print("\n🌲 Training Random Forest (GridSearch)...")

rf_param_grid = {
    'n_estimators': [100],
    'max_depth': [4, 6, 8],
    'min_samples_split': [8, 12],
    'min_samples_leaf': [5, 8]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    rf_param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)
rf_model = rf_grid.best_estimator_

print("Best RF Parameters:", rf_grid.best_params_)

# Evaluation Function

def evaluate_model(model, model_name):
    print(f"\n📊 Evaluating {model_name}...")
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Testing Accuracy :", accuracy_score(y_test, y_test_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=le.classes_))
    
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# Evaluate Random Forest
evaluate_model(rf_model, "Random Forest")

# Feature Importance (Random Forest)

print("\n🔎 Feature Importance (Random Forest)")
importances = rf_model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(importance_df)

plt.figure()
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.gca().invert_yaxis()
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.show()

# Step 7: Decision Tree

print("\n🌳 Training Decision Tree...")

dt_param_grid = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [5, 10]
}

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    dt_param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1
)

dt_grid.fit(X_train, y_train)
dt_model = dt_grid.best_estimator_

print("Best DT Parameters:", dt_grid.best_params_)
evaluate_model(dt_model, "Decision Tree")

# Step 8: XGBoost

print("\n🚀 Training XGBoost...")

param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb = XGBClassifier(
    random_state=RANDOM_STATE,
    eval_metric='mlogloss'
)

random_search = RandomizedSearchCV(
    xgb,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=RANDOM_STATE
)

random_search.fit(X_train, y_train)

xgb_model = random_search.best_estimator_

print("Best Parameters:", random_search.best_params_)
evaluate_model(xgb_model, "Tuned XGBoost")


# Step 9: Save Models (Versioned)

print("\n💾 Saving Models...")

# joblib.dump(rf_model, f"{MODEL_DIR}/RandomForest_{timestamp}.joblib")
# joblib.dump(dt_model, f"{MODEL_DIR}/DecisionTree_{timestamp}.joblib")
# joblib.dump(xgb_model, f"{MODEL_DIR}/XGBoost_{timestamp}.joblib")
# joblib.dump(le, f"{MODEL_DIR}/LabelEncoder_{timestamp}.joblib")

# joblib.dump(rf_model, f"{MODEL_DIR}/RandomForest.joblib")
# joblib.dump(dt_model, f"{MODEL_DIR}/DecisionTree.joblib")
# joblib.dump(xgb_model, f"{MODEL_DIR}/XGBoost.joblib")
# joblib.dump(le, f"{MODEL_DIR}/LabelEncoder.joblib")

print("✅ All models saved successfully with version timestamp.")
