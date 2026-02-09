import joblib 
import numpy as np 
import os 

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, 'iris_model.joblib'))
    return model 

def predict_fn(data, model):
    data = np.array(data)
    return model.predict(data)