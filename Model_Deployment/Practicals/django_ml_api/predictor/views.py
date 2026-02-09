import os
import joblib 
import numpy as np 
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response 

# Create your views here.
model_path = os.path.join(settings.BASE_DIR, "model.joblib")
try:
    model = joblib.load(model_path)
except:
    try:
        model = joblib.load('model.joblib')
    except:
        model = None 

class_names = ['setosa', 'versicolor', 'virginica']

@api_view(['GET'])
def health_view(request):
    return Response({'status': "Django app is running"})

@api_view(['POST'])
def predict_view(request):
    features = request.data.get('features')
    features_arr = np.array(features).reshape(1, -1)

    prediction = model.predict(features_arr)[0]
    probability = model.predict_proba(features_arr).max()

    return Response({
        'prediction': int(prediction),
        'class_name': class_names[int(prediction)],
        'confidence': float(probability)
    })

