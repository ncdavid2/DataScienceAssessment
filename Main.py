from calendar import Month

import pandas as pd

# https://healthdata.gov/dataset/National-Survey-of-Children-s-Health-NSCH-Vision-a/sw29-8nx8/about_data

# Pfad zur CSV-Datei im Downloads-Ordner
file_path = 'National_Survey_of_Children_s_Health__NSCH____Vision_and_Eye_Health_Surveillance.csv'

# CSV-file reader
df = pd.read_csv(file_path)

# Information shown expansion
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# return the first few rows
print(df.head())

# shows how many columns are in the file
print(df.count().sum())

# shows how any columns in the file are null
print(df.isnull().sum().sum())

