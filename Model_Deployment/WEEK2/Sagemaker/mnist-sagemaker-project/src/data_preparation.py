"""
MNIST Data Preparation Script
Downloads, preprocesses, and saves MNIST data for training.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle

# Configuration
DATA_DIR = './data'
BATCH_SIZE = 64
NUM_WORKERS = 2

def get_transforms():
    """Define data transformations."""
    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    return train_transform, test_transform

def download_and_prepare_data():
    """Download MNIST and create data loaders."""
    print('Downloading and preparing MNIST dataset...')
    
    train_transform, test_transform = get_transforms()
    
    # Download datasets
    train_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Test samples: {len(test_dataset)}')
    print(f'Number of classes: {len(train_dataset.classes)}')
    
    return train_loader, test_loader, train_dataset, test_dataset

def save_sample_data(test_dataset, num_samples=100):
    """Save sample data for later testing."""
    samples = []
    labels = []
    
    for i in range(num_samples):
        img, label = test_dataset[i]
        samples.append(img.numpy())
        labels.append(label)
    
    sample_data = {
        'images': np.array(samples),
        'labels': np.array(labels)
    }
    
    os.makedirs('data/samples', exist_ok=True)
    with open('data/samples/test_samples.pkl', 'wb') as f:
        pickle.dump(sample_data, f)
    
    print(f'Saved {num_samples} test samples to data/samples/')

if __name__ == '__main__':
    train_loader, test_loader, train_ds, test_ds = download_and_prepare_data()
    save_sample_data(test_ds)
    print('Data preparation complete!')

