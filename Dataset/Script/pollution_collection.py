import pandas as pd
import os

# Folder where all city CSV files are stored
input_folder = "city_pollution"
output_file = "Main_Pollution_Dataset.csv"

# Pollutants we need
required_params = ['pm25', 'pm10', 'no2', 'co', 'so2', 'o3']

main_df = pd.DataFrame()

for file in os.listdir(input_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(input_folder, file)
        
        df = pd.read_csv(file_path)
        
        # Keep only needed columns
        df = df[['location_name', 'latitude', 'longitude', 'parameter', 'value', 'datetimeUtc']]
        
        # Filter required pollutants
        df = df[df['parameter'].isin(required_params)]
        
        # Pivot table
        df_pivot = df.pivot_table(
            index=['location_name', 'latitude', 'longitude', 'datetimeUtc'],
            columns='parameter',
            values='value',
            aggfunc='mean'
        ).reset_index()
        
        # Rename date column
        df_pivot.rename(columns={'datetimeUtc': 'timestamp'}, inplace=True)
        df_pivot.rename(columns={'location_name':'city'}, inplace=True)
        
        # Append to main dataframe
        main_df = pd.concat([main_df, df_pivot], ignore_index=True)

# Save final dataset
main_df.to_csv(output_file, index=False)

print("Main dataset created successfully!")
