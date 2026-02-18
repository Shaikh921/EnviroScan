import pandas as pd

# Load Dataset

df = pd.read_csv("Final_Dataset_Cleaned.csv")

# print("File Loaded Successfully")
# print("Initial Shape:", df.shape)
# print("Duplicate Rows:", df.duplicated().sum())
# print("\nInitial Missing Percentage:")
# print((df.isnull().sum() / len(df)) * 100)

# Convert Datetime
df['datetimeUtc'] = pd.to_datetime(df['datetimeUtc'])


#  Sort Properly (VERY IMPORTANT)

df = df.sort_values(['city', 'datetimeUtc'])


#  Pollution Columns

pollution_cols = ['co','no2','o3','pm10','pm25','so2']


#  Interpolate + Forward Fill + Backward Fill

# df[pollution_cols] = (
#     df.groupby('city')[pollution_cols]
#       .transform(lambda x: x.interpolate(method='linear')
#                                .ffill()
#                                .bfill())
# )

#  Final Check

# print("\nAfter Cleaning Shape:", df.shape)
# print("\nRemaining Missing Values:")
# print(df.isnull().sum())

# Remove Remaining NaN value
df = df[df['city'] != 'Vikas Sadan, Gurugram - HSPCB']


# #  Clear Extrem Outliers

# for col in pollution_cols:
#     Q1 = df[col].quantile(0.25)
#     Q3 = df[col].quantile(0.75)
#     IQR = Q3 - Q1

#     lower = Q1 - 1.5 * IQR
#     upper = Q3 + 1.5 * IQR

#     df = df[(df[col] >= lower) & (df[col] <= upper)]


#  Clean Negative Values

# for col in pollution_cols:
#     df.loc[df[col] < 0, col] = None

# # Re-interpolate
# df[pollution_cols] = (
#     df.groupby('city')[pollution_cols]
#       .transform(lambda x: x.interpolate().ffill().bfill())
# )


#  Save Clean Dataset (Optional but Recommended)

# Check missing values per city
# print(
#     df.groupby('city')[['o3','pm25']]
#       .apply(lambda x: x.isnull().sum())
# )

print(df.describe())

df.to_csv("Final_Dataset_Cleaned1.csv")


print("\nCleaned dataset saved as Final_Dataset_Cleaned.csv")
