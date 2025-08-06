import requests
import json
import os
import time

# We need to test against the live container, so we'll use a hardcoded URL.
API_URL = "http://127.0.0.1:8000/predict"

def test_predict_endpoint():
    """
    Tests the /predict endpoint of the running API.
    """
    # Wait for the server to start. This is important for CI.
    print("Waiting for API server to start...")
    retries = 10
    while retries > 0:
        try:
            response = requests.post(API_URL, json={
                "V1": 0.0, "V2": 0.0, "V3": 0.0, "V4": 0.0, "V5": 0.0, "V6": 0.0,
                "V7": 0.0, "V8": 0.0, "V9": 0.0, "V10": 0.0, "V11": 0.0, "V12": 0.0,
                "V13": 0.0, "V14": 0.0, "V15": 0.0, "V16": 0.0, "V17": 0.0, "V18": 0.0,
                "V19": 0.0, "V20": 0.0, "V21": 0.0, "V22": 0.0, "V23": 0.0, "V24": 0.0,
                "V25": 0.0, "V26": 0.0, "V27": 0.0, "V28": 0.0, "Amount": 0.0
            })
            if response.status_code == 200:
                print("API server is up!")
                break
        except requests.exceptions.ConnectionError:
            time.sleep(1)
            retries -= 1
    assert retries > 0, "API server did not start in time."

    # Test the actual prediction endpoint
    sample_data = {
        "V1": -1.3598071336738, "V2": -0.0727811732551, "V3": 2.5362473405167,
        "V4": 1.37815522427448, "V5": -0.33832076994255, "V6": 0.46238777776269,
        "V7": 0.23959855490417, "V8": 0.09869790126105, "V9": 0.36378696961121,
        "V10": 0.09079417189317, "V11": -0.55159953326081, "V12": -0.61780085576182,
        "V13": -0.99138984723659, "V14": -0.31116935369987, "V15": 1.46817614013149,
        "V16": -0.47040052525949, "V17": 0.20797124192924, "V18": 0.02579058019855,
        "V19": 0.4039930322765, "V20": 0.25141209823979, "V21": -0.01830677793131,
        "V22": 0.27783757555889, "V23": -0.11047391018876, "V24": 0.06692807491465,
        "V25": 0.12853935827352, "V26": -0.18911484384993, "V27": 0.13355837674039,
        "V28": -0.02105305344538, "Amount": 149.62
    }
    response = requests.post(API_URL, json=sample_data)

    assert response.status_code == 200
    assert response.json() == {"prediction": "Legitimate", "class": 0}