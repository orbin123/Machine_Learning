from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("model.joblib")

@app.post("/predict")
def predict():
    features = request.json["features"]
    prediction = model.predict([features])[0]
    return jsonify({'predicition': str(prediction)})

app.run(host="0.0.0.0", port=5000)

