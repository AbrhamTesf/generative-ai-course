import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib

def preprocess_data():
    """
    Loads processed data, preprocesses it, and saves the split datasets.
    """
    print("Starting data preprocessing...")

    # Define paths
    script_dir = os.path.dirname(__file__)
    processed_dir = os.path.join(script_dir, '..', 'data', 'processed')
    processed_file_path = os.path.join(processed_dir, 'creditcard.csv')

    # Load data from the processed directory
    try:
        df = pd.read_csv(processed_file_path)
        print("Processed data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The processed data file was not found at {processed_file_path}")
        return

    # Check for missing values (This dataset is clean, but it's good practice to check)
    if df.isnull().sum().any():
        print("Warning: Missing values found. Handling not implemented in this version.")

    # Drop irrelevant columns (The 'Time' column is a relative time, not critical for the model)
    df = df.drop('Time', axis=1)

    # Separate features (X) and labels (y)
    X = df.drop('Class', axis=1)
    y = df['Class']
    print(f"Data shape before splitting: X={X.shape}, y={y.shape}")

    # Scale the 'Amount' feature using StandardScaler
    scaler = StandardScaler()
    X['Amount'] = scaler.fit_transform(X[['Amount']])
    print("'Amount' feature normalized successfully.")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Data split into training (80%) and testing (20%) sets.")
    print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")

    # Save the split datasets and the scaler to disk
    joblib.dump(X_train, os.path.join(processed_dir, 'X_train.joblib'))
    joblib.dump(X_test, os.path.join(processed_dir, 'X_test.joblib'))
    joblib.dump(y_train, os.path.join(processed_dir, 'y_train.joblib'))
    joblib.dump(y_test, os.path.join(processed_dir, 'y_test.joblib'))
    joblib.dump(scaler, os.path.join(processed_dir, 'scaler.joblib'))
    print("Split datasets and scaler saved to disk.")

if __name__ == "__main__":
    preprocess_data()