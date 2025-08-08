import pandas as pd
import json
import re
import warnings

warnings.filterwarnings('ignore')

def parse_logs(log_file_path):
    """Parses a log file and extracts prediction logs."""
    prediction_logs = []
    
    try:
        with open(log_file_path, 'r') as file:
            for line in file:
                # Use a regex to find the JSON-like object in each line
                match = re.search(r"(\{.*\})", line)
                if match:
                    json_str = match.group(1)
                    # Replace single quotes with double quotes for valid JSON
                    json_str = json_str.replace("'", '"')
                    try:
                        log_entry = json.loads(json_str)
                        prediction_logs.append(log_entry)
                    except json.JSONDecodeError as e:
                        print(f"Skipping malformed JSON entry: {e} in line: {line.strip()}")
                        continue
    except FileNotFoundError:
        print(f"Error: The file '{log_file_path}' was not found. Make sure the file exists and the path is correct.")
        return None
    
    return prediction_logs

def analyze_and_report():
    """Analyzes prediction logs for data drift and generates a report."""
    log_file = 'logs.txt'
    
    print("Parsing prediction logs...")
    
    logs = parse_logs(log_file)
    
    if logs is None or not logs:
        print("No prediction logs found to analyze.")
        return

    df = pd.DataFrame(logs)

    # Convert 'input_data' dictionary to separate columns
    input_data_df = df['input_data'].apply(pd.Series)
    df = pd.concat([df.drop('input_data', axis=1), input_data_df], axis=1)

    print(f"Found {len(df)} prediction logs.")

    # Drop non-numeric columns for analysis
    numeric_df = df.drop(columns=['timestamp', 'confidence'])

    # Calculate mean and standard deviation
    report = numeric_df.agg(['mean', 'std']).transpose()
    report.columns = ['Mean', 'Standard Deviation']

    # Identify potential drift in 'Amount' and 'V10' as examples
    print("\nData Drift Report:")
    print("-" * 25)

    initial_mean_amount = df.loc[df.index < 10, 'Amount'].mean()
    current_mean_amount = df.loc[df.index >= 10, 'Amount'].mean()
    
    initial_mean_V10 = df.loc[df.index < 10, 'V10'].mean()
    current_mean_V10 = df.loc[df.index >= 10, 'V10'].mean()

    print(f"Initial Mean Amount: {initial_mean_amount:.2f}")
    print(f"Current Mean Amount: {current_mean_amount:.2f}")
    
    print("-" * 25)
    
    print(f"Initial Mean V10: {initial_mean_V10:.2f}")
    print(f"Current Mean V10: {current_mean_V10:.2f}")

    if abs(current_mean_amount - initial_mean_amount) > 100:
        print("\n⚠️  Alert: Significant drift detected in 'Amount' feature!")
    else:
        print("\n✅  No significant drift detected in 'Amount' feature.")

    if abs(current_mean_V10 - initial_mean_V10) > 10:
        print("\n⚠️  Alert: Significant drift detected in 'V10' feature!")
    else:
        print("\n✅  No significant drift detected in 'V10' feature.")

    print("\nFull Report:")
    print(report)

if __name__ == '__main__':
    analyze_and_report()
