"""
SageMaker Inference Script for MNIST Classifier

This script implements the required functions for SageMaker inference:
- model_fn: Load the model
- input_fn: Deserialize input data
- predict_fn: Run inference
- output_fn: Serialize output data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os


# Model architecture (must match training)
class MNISTClassifier(nn.Module):
    """CNN for MNIST digit classification."""
    
    def __init__(self, num_classes=10):
        super(MNISTClassifier, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=0)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def model_fn(model_dir):
    """
    Load model from the model_dir. Required by SageMaker.
    
    Args:
        model_dir: Directory where model artifacts are stored
        
    Returns:
        model: Loaded PyTorch model
    """
    print(f"Loading model from {model_dir}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model architecture
    model = MNISTClassifier()
    
    # Load model weights
    model_path = os.path.join(model_dir, "mnist_classifier.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    return model


def input_fn(request_body, request_content_type):
    """
    Deserialize input data.
    
    Args:
        request_body: The request payload
        request_content_type: Content type of the request
        
    Returns:
        Preprocessed input tensor
    """
    print(f"Content type: {request_content_type}")
    
    if request_content_type == "application/json":
        # Parse JSON input
        input_data = json.loads(request_body)
        
        # Handle different input formats
        if isinstance(input_data, dict):
            data = input_data.get("instances", input_data.get("data"))
        else:
            data = input_data
        
        # Convert to numpy array
        np_array = np.array(data, dtype=np.float32)
        
        # Ensure correct shape: (batch, channels, height, width)
        if len(np_array.shape) == 2:
            # Single image flattened: (784,) -> (1, 1, 28, 28)
            np_array = np_array.reshape(-1, 1, 28, 28)
        elif len(np_array.shape) == 3:
            # Single image: (1, 28, 28) -> (1, 1, 28, 28)
            np_array = np_array.reshape(-1, 1, 28, 28)
        elif len(np_array.shape) == 4:
            pass  # Already correct shape
        
        # Normalize if not already normalized
        if np_array.max() > 1.0:
            np_array = np_array / 255.0
        
        # Apply MNIST normalization
        np_array = (np_array - 0.1307) / 0.3081
        
        tensor = torch.tensor(np_array)
        return tensor
    
    elif request_content_type == "application/x-npy":
        # Handle numpy array input
        import io
        stream = io.BytesIO(request_body)
        np_array = np.load(stream, allow_pickle=True)
        return torch.tensor(np_array, dtype=torch.float32)
    
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """
    Run inference on the input data.
    
    Args:
        input_data: Preprocessed input tensor
        model: Loaded PyTorch model
        
    Returns:
        Prediction results
    """
    device = next(model.parameters()).device
    input_data = input_data.to(device)
    
    with torch.no_grad():
        output = model(input_data)
        probabilities = F.softmax(output, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
    
    return {
        "predictions": predictions.cpu().numpy().tolist(),
        "probabilities": probabilities.cpu().numpy().tolist()
    }


def output_fn(prediction, response_content_type):
    """
    Serialize prediction results.
    
    Args:
        prediction: Prediction results from predict_fn
        response_content_type: Desired response content type
        
    Returns:
        Serialized response
    """
    if response_content_type == "application/json":
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")
