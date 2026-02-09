
"""
Test Script for SageMaker Endpoint
"""

import boto3
import json
import numpy as np
import pickle

def load_test_samples():
    """Load saved test samples."""
    with open('../data/samples/test_samples.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['images'], data['labels']

def invoke_endpoint(endpoint_name, payload):
    """Invoke SageMaker endpoint."""
    runtime = boto3.client('sagemaker-runtime')
    
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps({'instances': payload.tolist()})
    )
    
    result = json.loads(response['Body'].read().decode())
    return result

def test_endpoint():
    """Run tests on the endpoint."""
    print('=' * 60)
    print('Testing SageMaker Endpoint')
    print('=' * 60)
    
    # Load endpoint name
    with open('endpoint_name.txt', 'r') as f:
        endpoint_name = f.read().strip()
    
    print(f'Endpoint: {endpoint_name}')
    print()
    
    # Load test data
    images, labels = load_test_samples()
    print(f'Loaded {len(images)} test samples')
    print()
    
    # Test single prediction
    print('Testing single prediction...')
    single_image = images[0:1]  # Shape: (1, 1, 28, 28)
    result = invoke_endpoint(endpoint_name, single_image)
    
    print(f'True label: {labels[0]}')
    print(f'Predicted: {result["predictions"][0]}')
    print(f'Confidence: {max(result["probabilities"][0]):.4f}')
    print()
    
    # Test batch prediction
    print('Testing batch prediction (10 samples)...')
    batch_images = images[0:10]
    batch_labels = labels[0:10]
    result = invoke_endpoint(endpoint_name, batch_images)
    
    predictions = result['predictions']
    correct = sum(p == l for p, l in zip(predictions, batch_labels))
    
    print(f'Results:')
    print(f'  True labels:  {batch_labels.tolist()}')
    print(f'  Predictions:  {predictions}')
    print(f'  Accuracy: {correct}/{len(batch_labels)} ({100*correct/len(batch_labels):.1f}%)')
    print()
    
    # Test larger batch
    print('Testing larger batch (100 samples)...')
    result = invoke_endpoint(endpoint_name, images)
    predictions = result['predictions']
    correct = sum(p == l for p, l in zip(predictions, labels))
    
    print(f'  Accuracy: {correct}/{len(labels)} ({100*correct/len(labels):.1f}%)')
    print()
    print('=' * 60)
    print('Tests Complete!')
    print('=' * 60)

if __name__ == '__main__':
    test_endpoint()
