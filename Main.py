import pandas as pd

# https://healthdata.gov/dataset/National-Survey-of-Children-s-Health-NSCH-Vision-a/sw29-8nx8/about_data

df = pd.read_csv('National_Survey_of_Children_s_Health__NSCH____Vision_and_Eye_Health_Surveillance.csv')

# Information shown expansion
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# return the first few rows
print(df.head())
# shows how many columns are in the file
print(df.count().sum())
# shows how any columns in the file are null
print(df.isnull().sum().sum())
