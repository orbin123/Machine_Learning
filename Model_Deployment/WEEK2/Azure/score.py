import joblib
import numpy as np
import os

def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "iris_model.joblib")
    model = joblib.load(model_path)

def run(data):
    data = np.array(data)
    predictions = model.predict(data)
    return predictions.tolist()