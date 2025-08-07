import logging
from datetime import datetime

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
        "prediction": prediction,
        "confidence": confidence,
    }
    logging.info(log_entry)