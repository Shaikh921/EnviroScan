# ==============================
# EnviroScan - Week 4
# Step 7: Feature Importance
# ==============================

import pandas as pd
import matplotlib.pyplot as plt
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

# Train Best Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    random_state=42
)

rf.fit(X_train, y_train)

# Get Feature Importance
importances = rf.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
})

importance_df = importance_df.sort_values(by='Importance', ascending=False)

print(importance_df)

# Plot
plt.figure()
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel("Importance Score")
plt.title("Feature Importance - Random Forest")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()