import pandas as pd  
import matplotlib.pyplot as plt
import numpy as np


# Loading Dataset
df = pd.read_csv("Dataset/Final_Dataset_Cleaned.csv")
print("Dataset Loaded Successfully")
print("Initial Shape:", df.shape)


# Define Percentile-Based Pollutant Thresholds
# (Using 65th percentile for better class balance)


no2_high = df['no2'].quantile(0.65)
pm25_high = df['pm25'].quantile(0.65)
pm10_high = df['pm10'].quantile(0.65)
so2_high = df['so2'].quantile(0.65)
co_high = df['co'].quantile(0.65)

print("\nThreshold Values:")
print("NO2:", no2_high)
print("PM2.5:", pm25_high)
print("PM10:", pm10_high)
print("SO2:", so2_high)
print("CO:", co_high)

#  Define Distance Thresholds (Meters)


road_close = 600
industry_close = 2000
dump_close = 6000
farm_close = 4000

def assign_label(row):
    # Industrial
    if row['so2'] > so2_high and row['dist_to_industry'] < industry_close:
        return "Industrial"

    # Vehicular
    elif row['no2'] > no2_high and row['dist_to_road'] < road_close:
        return "Vehicular"

    # Burning
    elif row['pm25'] > pm25_high and row['co'] > co_high and row['dist_to_dump'] < dump_close:
        return "Burning"

    # Agricultural
    elif row['pm10'] > pm10_high and row['dist_to_farmland'] < farm_close and row['Humidity'] < 70:
        return "Agricultural"

    # Natural
    else:
        return "Natural"


df['pollution_source'] = df.apply(assign_label, axis=1)
print("\nLabeling Completed Successfully")


# Class Distribution

distribution = df['pollution_source'].value_counts()
percentage = df['pollution_source'].value_counts(normalize=True) * 100

print("\nClass Distribution (Count):")
print(distribution)

print("\nClass Distribution (%):")
print(percentage.round(2))

#  Distribution in Graph Form
distribution.plot(kind='bar')

plt.title("Balanced Pollution Source Distribution")
plt.xlabel("Pollution Source")
plt.ylabel("Number of Records")
plt.xticks(rotation=45)

# Set Y-axis interval to 500
plt.yticks(np.arange(0, distribution.max() + 500, 500))

plt.tight_layout()
plt.show()


#  Save Final Labeled Dataset

df.to_csv("Final_Dataset_Labeled_Balanced.csv", index=False)

print("\nFinal Labeled Dataset Saved Successfully!")
print("File Name: Final_Dataset_Labeled_Balanced.csv")