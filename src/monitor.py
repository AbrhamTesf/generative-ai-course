import logging
from datetime import datetime
import json

# Configure logging to write to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_prediction(input_data, prediction, confidence=None):
    """Logs the details of a prediction request and its result."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input_data": input_data,
        "prediction": int(prediction),
        "confidence": float(confidence) if confidence is not None else None,
    }
    # Log the JSON object as a string
    logging.info(json.dumps(log_entry))