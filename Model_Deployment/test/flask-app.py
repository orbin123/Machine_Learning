from flask import Flask, jsonify, request 
import numpy as np 
import joblib 

app = Flask(__name__)

model = joblib.load('model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.to_json()

    data = np.array(data).reshape(-1, 1)

    prediction = model.predict(data)[0]

    return jsonify({'prediction': prediction})

if __name__=='__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)