"""
SageMaker Deployment Script

Deploys the MNIST classifier to a SageMaker endpoint.
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
import time
import os

# Configuration
CONFIG = {
    'model_name': 'mnist-classifier',
    'endpoint_name': 'mnist-classifier-endpoint',
    'instance_type': 'ml.t2.medium',  # Free tier eligible
    'instance_count': 1,
    'framework_version': '2.1.0',
    'py_version': 'py310',
}

def get_role_arn():
    """Get the SageMaker execution role ARN."""
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='SageMakerExecutionRole')
    return role['Role']['Arn']

def get_model_uri():
    """Construct the S3 URI for the model."""
    sts = boto3.client('sts')
    account_id = sts.get_caller_identity()['Account']
    region = boto3.session.Session().region_name
    bucket = f'sagemaker-mnist-{account_id}-{region}'
    return f's3://{bucket}/models/mnist/model.tar.gz'

def deploy_model():
    """Deploy model to SageMaker endpoint."""
    print('=' * 60)
    print('SageMaker Deployment')
    print('=' * 60)
    
    # Get configuration
    role_arn = get_role_arn()
    model_uri = get_model_uri()
    
    print(f'Role ARN: {role_arn}')
    print(f'Model URI: {model_uri}')
    print(f'Instance Type: {CONFIG["instance_type"]}')
    print()
    
    # Create SageMaker session
    session = sagemaker.Session()
    
    # Create PyTorch model
    print('Creating SageMaker model...')
    pytorch_model = PyTorchModel(
        model_data=model_uri,
        role=role_arn,
        entry_point='inference.py',
        source_dir=None,  # inference.py is in code/ inside model.tar.gz
        framework_version=CONFIG['framework_version'],
        py_version=CONFIG['py_version'],
        name=f"{CONFIG['model_name']}-{int(time.time())}",
        sagemaker_session=session,
    )
    
    # Deploy to endpoint
    print('\nDeploying to endpoint (this takes 5-10 minutes)...')
    print('Please wait...')
    
    predictor = pytorch_model.deploy(
        initial_instance_count=CONFIG['instance_count'],
        instance_type=CONFIG['instance_type'],
        endpoint_name=f"{CONFIG['endpoint_name']}-{int(time.time())}",
        wait=True,
    )
    
    endpoint_name = predictor.endpoint_name
    
    print('\n' + '=' * 60)
    print('Deployment Successful!')
    print('=' * 60)
    print(f'Endpoint Name: {endpoint_name}')
    print()
    print('Save this endpoint name for testing!')
    
    # Save endpoint name to file
    with open('endpoint_name.txt', 'w') as f:
        f.write(endpoint_name)
    
    return endpoint_name

def delete_endpoint(endpoint_name):
    """Delete the SageMaker endpoint to stop charges."""
    print(f'Deleting endpoint: {endpoint_name}')
    
    sm_client = boto3.client('sagemaker')
    
    # Delete endpoint
    sm_client.delete_endpoint(EndpointName=endpoint_name)
    print('Endpoint deleted.')
    
    # Wait for deletion
    print('Waiting for endpoint to be deleted...')
    waiter = sm_client.get_waiter('endpoint_deleted')
    waiter.wait(EndpointName=endpoint_name)
    print('Endpoint deletion complete.')

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'delete':
        if len(sys.argv) > 2:
            delete_endpoint(sys.argv[2])
        else:
            with open('endpoint_name.txt', 'r') as f:
                endpoint_name = f.read().strip()
            delete_endpoint(endpoint_name)
    else:
        deploy_model()
