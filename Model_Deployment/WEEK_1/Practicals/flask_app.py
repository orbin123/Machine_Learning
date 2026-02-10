from flask import Flask, request, jsonify 
import joblib 
import numpy as np 

# source /Users/orbinsunny/Documents/GitHub/Machine_Learning/Web-Frameworks/Flask/venv/bin/activate

model = joblib.load('model.joblib')

class_names = ['setosa', 'versicolor', 'virginica']

app = Flask(__name__)

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'service': "Flask ML Inference",
        'status': 'running',
        'model_loaded': True,
        'endpoints': {
            'predict':'/predict',
            'health': '/'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    features = data.get('features')

    features_array = np.array(features).reshape(1, -1)

    prediction = model.predict(features_array)[0]

    probabilities = model.predict_proba(features_array)[0]

    confidence = float(probabilities[prediction])

    print(class_names[prediction])

    return jsonify({
        'prediction': int(prediction),
        'class_name': class_names[prediction],
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=8080,
        debug=True 
    )