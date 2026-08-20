
# 
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt


# df = pd.read_csv("Dataset/Final_Dataset_Labeled_Balanced.csv")

# df = df.drop(columns = ["Unnamed: 0"], errors='ignore')
# numeric_df = df.select_dtypes(include=['int64', 'float64'])

# plt.figure(figsize = (12,8))
# sns.heatmap(numeric_df.corr(),annot=True, cmap = "coolwarm",fmt = ".2f")

# plt.title("Correlation Heatmap - EnviroScan Dataset")
# plt.xticks(rotation=45)
# plt.yticks(rotation=0)

# plt.tight_layout()
# plt.show()
# # print("Hello")

# # Histogram

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load dataset
# df = pd.read_csv("Dataset/Final_Dataset_Labeled_Balanced.csv")

# # Drop unnecessary column
# df = df.drop(columns=["Unnamed: 0"], errors='ignore')

# # Select important numerical columns for plotting
# features = ['pm25', 'pm10', 'co', 'no2', 'o3', 'Temperature']

# # Set style
# sns.set_style("whitegrid")

# # Create histograms with distribution (KDE)
# for feature in features:
#     plt.figure(figsize=(8,5))
#     sns.histplot(df[feature], kde=True, bins=30)
    
#     plt.title(f"Distribution of {feature}")
#     plt.xlabel(feature)
#     plt.ylabel("Frequency")
    
#     plt.tight_layout()
#     plt.show()
# 3.3 Boxplot
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load dataset
# df = pd.read_csv("Dataset/Final_Dataset_Labeled_Balanced.csv")

# # Drop unnecessary column
# df = df.drop(columns=["Unnamed: 0"], errors='ignore')

# # Select numerical columns
# numeric_df = df.select_dtypes(include=['int64', 'float64'])

# # Plot all boxplots in one figure
# plt.figure(figsize=(14, 8))
# sns.boxplot(data=numeric_df)

# plt.title("Boxplot of All Numerical Features")
# plt.xticks(rotation=45)

# plt.tight_layout()
# plt.show()
# CountPLot
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load dataset
# df = pd.read_csv("Dataset/Final_Dataset_Labeled_Balanced.csv")

# # Drop unnecessary column
# df = df.drop(columns=["Unnamed: 0"], errors='ignore')

# # Set style
# sns.set_style("whitegrid")

# # Create subplots
# fig, axes = plt.subplots(1, 2, figsize=(16,6))

# # Countplot for city
# sns.countplot(y='city', data=df, order=df['city'].value_counts().index, ax=axes[0])
# axes[0].set_title("Count of Records by City")

# # Countplot for pollution_source
# sns.countplot(x='pollution_source', data=df, order=df['pollution_source'].value_counts().index, ax=axes[1])
# axes[1].set_title("Count of Pollution Sources")

# plt.tight_layout()
# plt.show()
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load dataset
# df = pd.read_csv("Dataset/Final_Dataset_Labeled_Balanced.csv")

# # Drop unnecessary column
# df = df.drop(columns=["Unnamed: 0"], errors='ignore')

# # Select important features (keep it limited for clarity)
# features = ['pm25', 'pm10', 'co', 'no2', 'Temperature']

# # Take sample (important for speed)
# sample_df = df[features].sample(n=500, random_state=42)

# # Create pairplot
# sns.pairplot(sample_df, diag_kind='kde')

# plt.suptitle("Pairplot of Key Environmental Features", y=1.02)

# plt.show()


# 1. Actual vs Predicted Graph (Figure 6.1)
# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# 1. LOAD DATASET
# -----------------------------
df = pd.read_csv("Dataset/Final_Dataset_Labeled_Balanced.csv")

print("Dataset Loaded Successfully!\n")
print(df.head())

# -----------------------------
# 2. PREPROCESSING
# -----------------------------

# Drop unwanted column
df = df.drop(columns=["Unnamed: 0"], errors='ignore')

# Convert datetime
df['datetimeUtc'] = pd.to_datetime(df['datetimeUtc'])
df['datetime_hour'] = pd.to_datetime(df['datetime_hour'])

# Feature extraction
df['year'] = df['datetimeUtc'].dt.year
df['month'] = df['datetimeUtc'].dt.month
df['day'] = df['datetimeUtc'].dt.day
df['hour'] = df['datetime_hour'].dt.hour

# Drop original datetime columns
df = df.drop(columns=['datetimeUtc', 'datetime_hour'])

# Fill missing values (FIX WARNING)
df = df.ffill()

# -----------------------------
# 3. FEATURE & TARGET (CORRECT)
# -----------------------------

# Separate target BEFORE encoding
y = df['pollution_source']

# Features
X = df.drop('pollution_source', axis=1)

# Encode ONLY features
X = pd.get_dummies(X, drop_first=True)

# -----------------------------
# 4. TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. MODEL TRAINING
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 6. PREDICTION
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 7. EVALUATION
# -----------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# 8. CORRELATION HEATMAP
# -----------------------------
plt.figure(figsize=(12, 8))
sns.heatmap(X.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# 9. FEATURE IMPORTANCE
# -----------------------------
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.show()