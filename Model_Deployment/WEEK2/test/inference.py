import numpy as np
import joblib 

def model_fn(model_path):
    model = joblib.load('iris_model.joblib')
    return model 

def predict_fn(data, model):
    data = np.array(data)
    return model.predict(data)