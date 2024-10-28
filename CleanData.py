import pandas as pd

# Load the CSV file
df = pd.read_csv('National_Survey_of_Children_s_Health__NSCH____Vision_and_Eye_Health_Surveillance.csv')

# Print the column names before deletion
print("Columns before deletion:")
print(df.columns)

# Delete a specific column
columns_to_delete = ['Data_Value', 'Data_Value_Footnote_Symbol', 'Data_Value_Footnote', 'Low_Confidence_Limit']
df = df.drop(columns=columns_to_delete)

# Print the column names after deletion
print("\nColumns after deletion:")
print(df.columns)

# Save the updated DataFrame back to a CSV file (optional)
df.to_csv('updated_NSCH_Vision_Health_Data.csv', index=False)