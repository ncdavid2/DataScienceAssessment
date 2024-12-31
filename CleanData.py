import pandas as pd

df = pd.read_csv('National_Survey_of_Children_s_Health__NSCH____Vision_and_Eye_Health_Surveillance.csv')

print("Columns before deletion:")
print(df.columns)
columns_to_delete = ['Data_Value_Footnote_Symbol', 'Data_Value_Footnote', 'Low_Confidence_Limit', 'TopicID', 'CategoryID', 'QuestionID', 'ResponseID',
                     'AgeID', 'GenderID', 'Data_Value_Type', 'High_Confidence_Limit', 'Numerator', 'DataValueTypeID', 'GeoLocation', 'Geographic Level',
                     'RaceEthnicityID', 'RiskFactorID', 'RiskFactorResponseID', 'Datasource', 'RiskFactor', 'RiskFactorResponse', 'LocationID']

df = df.drop(columns=columns_to_delete)

df.dropna(subset=['Sample_Size'], inplace=True)

print("\nColumns after deletion:")
print(df.columns)

df['Data_Value'] = df['Data_Value'].fillna(0)

# Save the updated DataFrame back to a CSV file
df.to_csv('updated_NSCH_Vision_Health_Data.csv', index=False)

