import pandas as pd
import os

def save_data_to_processed():
    """
    Loads the raw data and saves it to the processed directory.
    """
    # Define paths
    script_dir = os.path.dirname(__file__)
    raw_file_path = os.path.join(script_dir, '..', 'data', 'raw', 'creditcard.csv')
    processed_dir = os.path.join(script_dir, '..', 'data', 'processed')
    processed_file_path = os.path.join(processed_dir, 'creditcard.csv')

    # Ensure the processed directory exists
    os.makedirs(processed_dir, exist_ok=True)

    # Load the raw data
    try:
        df = pd.read_csv(raw_file_path)
        print("Raw data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The raw data file was not found at {raw_file_path}")
        return

    # Save the data to the processed directory
    try:
        df.to_csv(processed_file_path, index=False)
        print(f"Data saved successfully to {processed_file_path}")
    except Exception as e:
        print(f"An error occurred while saving the data: {e}")

if __name__ == "__main__":
    save_data_to_processed()