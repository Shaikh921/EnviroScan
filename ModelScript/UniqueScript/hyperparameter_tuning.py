# ==============================
# EnviroScan - Week 4
# Step 6: Hyperparameter Tuning (Random Forest)
# ==============================

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
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

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

# Grid Search
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

# Best model
best_rf = grid.best_estimator_

# Evaluate on test
y_test_pred = best_rf.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_test_pred))