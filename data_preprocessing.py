import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_data():
    """Load and preprocess the heart disease dataset"""
    print("Checking for data file...")
    data_file = 'data/heart_disease.csv'
    
    # Check if file exists
    if os.path.exists(data_file):
        print(f"Found data file at: {os.path.abspath(data_file)}")
        try:
            df = pd.read_csv(data_file)
            print(f"Successfully loaded data: {len(df)} records")
            return df
        except Exception as e:
            print(f"Error loading data file: {e}")
    
    print("Data file not found locally. Downloading from UCI repository...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Load dataset from UCI repository
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    try:
        print("Downloading data from UCI repository...")
        df = pd.read_csv(url, names=column_names, na_values='?')
        print(f"Downloaded {len(df)} records")
        
        # Save a local copy
        df.to_csv(data_file, index=False)
        print(f"Saved data to {os.path.abspath(data_file)}")
        
        return df
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None
    
    return df

def preprocess_data(df):
    """Preprocess the data"""
    # Convert target to binary (0 = no disease, 1 = disease)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    # Handle missing values
    df['ca'] = df['ca'].fillna(df['ca'].mode()[0])
    df['thal'] = df['thal'].fillna(df['thal'].mode()[0])
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()

if __name__ == "__main__":
    # Test data loading and preprocessing
    df = load_data()
    if df is not None:
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
        print(df.head())
        print(f"\nTraining set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Feature names: {feature_names}")
