from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the logistic regression model pipeline
model = joblib.load('lgbmreg_model.joblib')


# Define the input data for model
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: str
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str 
    MultipleLines: str 
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str 
    TechSupport: str
    StreamingTV: str 
    StreamingMovies: str 
    Contract: str
    PaperlessBilling: str 
    PaymentMethod: str
    MonthlyCharges: str
    TotalCharges: str


# Create FastAPI app
app = FastAPI()

# Define prediction endpoint
@app.post("/predict")
def predict(data: CustomerData):
    # Convert input data to a dictionary and then to a DataFrame
    input_data = {
        'gender': [data.gender],
        'SeniorCitizen': [data.SeniorCitizen],
        'Partner': [data.Partner],
        'Dependents': [data.Dependents],
        'tenure': [data.tenure],
        'PhoneService': [data.PhoneService],
        'MultipleLines': [data.MultipleLines],
        'InternetService': [data.InternetService],
        'OnlineSecurity': [data.OnlineSecurity],
        'OnlineBackup': [data.OnlineBackup],
        'DeviceProtection': [data.DeviceProtection],
        'TechSupport': [data.TechSupport],
        'StreamingTV': [data.StreamingTV],
        'StreamingMovies': [data.StreamingMovies],
        'Contract': [data.Contract],
        'PaperlessBilling': [data.PaperlessBilling],
        'PaymentMethod': [data.PaymentMethod],
        'MonthlyCharges': [data.MonthlyCharges],
        'TotalCharges': [data.TotalCharges]
    }

    import pandas as pd
    input_df = pd.DataFrame(input_data)

    # Make a prediction
    prediction = model.predict(input_df)

    # Return the prediction
    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# curl -X POST "http://localhost:8000/predict" \
# -H "Content-Type: application/json" \
# -d '{
#     "gender": "Female",
#     "SeniorCitizen": "0",
#     "Partner": "Yes",
#     "Dependents": "No",
#     "tenure": 24,
#     "PhoneService": "Yes",
#     "MultipleLines": "No",
#     "InternetService": "Fiber optic",
#     "OnlineSecurity": "No",
#     "OnlineBackup": "Yes",
#     "DeviceProtection": "No",
#     "TechSupport": "No",
#     "StreamingTV": "Yes",
#     "StreamingMovies": "No",
#     "Contract": "Month-to-month",
#     "PaperlessBilling": "Yes",
#     "PaymentMethod": "Electronic check",
#     "MonthlyCharges": "85.45",
#     "TotalCharges": "2052.80"
# }'