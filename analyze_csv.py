import pandas as pd
import ast

# Read just a few rows
df = pd.read_csv('2023_05_Swing_Routes.csv', nrows=5)
print(df['routes'].iloc[0])
try:
    data = ast.literal_eval(df['routes'].iloc[0])
    print("Parsed successfully with ast.literal_eval:")
    print(data[0]) 
except Exception as e:
    print(f"Failed to parse: {e}")
