from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

saved = joblib.load("iris_model.joblib")
model = saved["model"]
target_names = saved["target_names"]

@app.post("/predict")
def predict():
    features = request.get_json()["features"]

    prediction = model.predict([features])[0]
    return jsonify({"class": str(target_names[prediction])})

if __name__ == "__main__": 
    app.run(debug=True)