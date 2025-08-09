import pandas as pd
import os

def load_and_inspect_data():
    """
    Loads the raw credit card fraud dataset and prints key information.
    """
    # Define the path to the raw data file
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, '..', 'data', 'raw', 'creditcard.csv')

    # Load the data using pandas
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

    # Inspect the data
    print("\n--- Data Head ---")
    print(df.head())

    print("\n--- Data Info ---")
    df.info()

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Target Class Distribution ---")
    # The 'Class' column contains our target variable (0 for legitimate, 1 for fraudulent)
    class_distribution = df['Class'].value_counts()
    print(class_distribution)
    print(f"\nLegitimate transactions: {class_distribution.get(0, 0)} ({class_distribution.get(0, 0)/len(df)*100:.2f}%)")
    print(f"Fraudulent transactions: {class_distribution.get(1, 0)} ({class_distribution.get(1, 0)/len(df)*100:.2f}%)")

    return df

if __name__ == "__main__":
    load_and_inspect_data()