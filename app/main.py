import joblib
import pandas as pd
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Define paths
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

# Load the trained model and scaler
try:
    model = joblib.load(os.path.join(models_dir, 'model.pkl'))
    scaler = joblib.load(os.path.join(processed_dir, 'scaler.joblib'))
    print("Model and scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: Required file not found. Have you trained the model? {e}")
    model, scaler = None, None

# Define the FastAPI application
app = FastAPI(title="Fraud Detection Model API")

# Define the Pydantic model for input data
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

# Define the prediction endpoint
@app.post("/predict", tags=["Prediction"])
async def predict(data: PredictionRequest):
    """
    Makes a fraud prediction based on the provided transaction data.
    """
    if model is None or scaler is None:
        return {"error": "Model or scaler not loaded. Please check your files."}

    # Convert the incoming data to a pandas DataFrame
    df = pd.DataFrame([data.model_dump()])
    
    # Preprocess the 'Amount' feature using the loaded scaler
    df['Amount'] = scaler.transform(df[['Amount']])
    
    # Make a prediction
    prediction = model.predict(df)
    
    # Return the prediction result
    if prediction[0] == 1:
        return {"prediction": "Fraudulent", "class": 1}
    else:
        return {"prediction": "Legitimate", "class": 0}