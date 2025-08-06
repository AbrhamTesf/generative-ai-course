import os
import joblib
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection Model API",
    version="0.1.0",
    description="Makes a fraud prediction based on the provided transaction data."
)

# Define the data model for the request body
class PredictionRequest(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# Load the trained model and scaler
# This function assumes the model and scaler are stored in a directory specified by the MODEL_PATH environment variable.
def load_model():
    model_path_base = os.getenv("MODEL_PATH", "./models")
    model_path = os.path.join(model_path_base, "model.pkl")
    scaler_path = os.path.join(model_path_base, "scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Model or scaler not found at: {model_path} or {scaler_path}")
        return None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model()

if model and scaler:
    print("Model and scaler loaded successfully.")
else:
    print("Error: Model or scaler not loaded. Please check your files.")
    model = None
    scaler = None

# API endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    if not model or not scaler:
        return {"error": "Model or scaler not loaded. Please check your files."}

    data = pd.DataFrame([request.dict()])
    
    # Scale the 'Amount' column
    data['Amount'] = scaler.transform(data[['Amount']])

    # Make prediction
    prediction = model.predict(data)[0]
    
    # Return result
    if prediction == 1:
        return {"prediction": "Fraudulent", "class": 1}
    else:
        return {"prediction": "Legitimate", "class": 0}