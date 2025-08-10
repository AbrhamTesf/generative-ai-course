import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load the dataset
try:
    df = pd.read_csv("data/raw/creditcard.csv")
except FileNotFoundError:
    print("creditcard.csv not found. Please ensure it's in the same directory.")
    exit()

# Drop the 'Time' column as it's not needed for the model
df = df.drop('Time', axis=1)

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Scale the 'Amount' column
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create the models directory if it doesn't exist
if not os.path.exists("models"):
    os.makedirs("models")

# Save the trained model and the scaler
joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model and scaler saved successfully in the 'models/' directory.")