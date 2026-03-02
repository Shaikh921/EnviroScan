# ==============================
# EnviroScan - Week 4
# Step 3: Encoding + Train-Test Split
# ==============================

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ------------------------------
# 1️⃣ Load Dataset
# ------------------------------
df = pd.read_csv("Dataset/Final_Dataset_Labeled.csv")

print("Dataset Loaded Successfully")
print("Dataset Shape:", df.shape)


# ------------------------------
# 2️⃣ Define Features (X)
# ------------------------------
X = df[['co','no2','o3','pm10','pm25','so2',
        'Temperature','Humidity','Wind Speed','Wind Direction',
        'dist_to_road','dist_to_industry','dist_to_dump','dist_to_farmland']]

# ------------------------------
# 3️⃣ Define Target (y)
# ------------------------------
y = df['pollution_source']

print("\nNumber of Features:", X.shape[1])
print("Number of Samples:", X.shape[0])


# ------------------------------
# 4️⃣ Encode Target Variable
# ------------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("\nClass Mapping:")
for index, class_name in enumerate(le.classes_):
    print(f"{class_name} --> {index}")


# ------------------------------
# 5️⃣ Train-Test Split (80-20)
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded   # keeps class balance
)

print("\nTrain-Test Split Completed")
print("Training Size:", X_train.shape)
print("Testing Size:", X_test.shape)


# ------------------------------
# 6️⃣ Check Class Distribution After Split
# ------------------------------
import numpy as np

print("\nTraining Class Distribution:")
print(np.bincount(y_train))

print("\nTesting Class Distribution:")
print(np.bincount(y_test))


print("\nStep 3 Completed Successfully ✅")