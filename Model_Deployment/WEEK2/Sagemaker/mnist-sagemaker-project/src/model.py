"""
CNN Model Architecture for MNIST Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTClassifier(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification.
    
    Architecture:
    - 3 Convolutional layers with batch normalization
    - 2 Fully connected layers
    - Dropout for regularization
    """
    
    def __init__(self, num_classes=10):
        super(MNISTClassifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=0
        )
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=0
        )
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=0
        )
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # After convolutions: 128 * 3 * 3 = 1152
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Conv block 1: 28x28 -> 26x26 -> 13x13
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv block 2: 13x13 -> 11x11 -> 5x5
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv block 3: 5x5 -> 3x3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Flatten: 128 * 3 * 3 = 1152
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def predict(self, x):
        """Get class predictions with probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        return predictions, probabilities


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model
    model = MNISTClassifier()
    print(f'Model architecture:\n{model}')
    print(f'\nTotal trainable parameters: {count_parameters(model):,}')
    
    # Test forward pass
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f'\nInput shape: {dummy_input.shape}')
    print(f'Output shape: {output.shape}')
