import pandas as pd
import requests

# API endpoint URL for your local environment
API_URL = "http://localhost:8000/predict"

# Load the original data (make sure this file exists in your data/ directory)
try:
    df = pd.read_csv("data/raw/creditcard.csv")
    df = df.drop(columns=['Time', 'Class'], errors='ignore')
except FileNotFoundError:
    print("Error: The 'data/raw/creditcard.csv' file was not found. Please ensure it's in the correct location.")
    exit()

# Take a sample of 100 transactions to simulate requests
sample_df = df.sample(n=100, random_state=42)

# --- Simulate Data Drift ---
# We'll simulate drift by adding a constant value to one of the features (e.g., V10)
# This changes the distribution of the data without altering its fundamental structure.
sample_df['V10'] = sample_df['V10'] + 50

# --- Send requests to the API ---
print("Sending drifted data to the API...")
drifted_predictions = []

for index, row in sample_df.iterrows():
    # Prepare the data as a JSON object
    data = row.to_dict()
    
    try:
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            drifted_predictions.append(response.json())
        else:
            print(f"Request failed with status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

print("Drift simulation complete.")
print(f"Received {len(drifted_predictions)} predictions.")