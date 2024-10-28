from calendar import Month

import pandas as pd

# https://healthdata.gov/dataset/National-Survey-of-Children-s-Health-NSCH-Vision-a/sw29-8nx8/about_data

# Pfad zur CSV-Datei im Downloads-Ordner
file_path = 'National_Survey_of_Children_s_Health__NSCH____Vision_and_Eye_Health_Surveillance.csv'

# CSV-Datei auslesen
df = pd.read_csv(file_path)

# Anzeigeeinstellungen erweitern
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Erste Zeilen anzeigen, um einen Überblick über die Daten zu bekommen
print(df.head())

# Zeige die Gesamte anzahl an Datensätzen an
print(df.count().sum())

# Zeige wieviele null Datensätze es gibt
print(df.isnull().sum().sum())

