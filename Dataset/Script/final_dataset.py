import pandas as pd

df_weather = pd.read_csv("Final_Pollution_Weather_Dataset.csv")
df_location = pd.read_csv("Location_Features_Dataset.csv")

merged = pd.merge(df_location,df_weather , on=["city", "latitude","longitude"])

merged.to_csv("Final_Dataset.csv",index=False)