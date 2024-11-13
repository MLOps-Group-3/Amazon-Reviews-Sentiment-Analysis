import pandas as pd

# Load the CSV file
df = pd.read_csv("/Users/vallimeenaa/Downloads/labeled_aspect_data.csv")

# Replace 'column_name' with the actual column name you're interested in
unique_values = df['aspect'].unique()

# Convert to a list if needed
unique_values_list = unique_values.tolist()

print(unique_values_list)
