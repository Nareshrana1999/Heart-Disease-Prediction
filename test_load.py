import pandas as pd
import os

try:
    print("Trying to load data file...")
    file_path = os.path.join('data', 'heart_disease_cleveland.csv')
    print(f"Looking for file at: {os.path.abspath(file_path)}")
    
    if os.path.exists(file_path):
        print("File exists!")
        df = pd.read_csv(file_path)
        print("Data loaded successfully!")
        print(f"Shape: {df.shape}")
        print("First few rows:")
        print(df.head())
    else:
        print("File does not exist.")
except Exception as e:
    print(f"Error: {str(e)}")
