
"""
Training Script for MNIST Classifier
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import os
from datetime import datetime

from model import MNISTClassifier, count_parameters
from data_preparation import download_and_prepare_data

# Training Configuration
CONFIG = {
    'epochs': 10,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'scheduler_step': 5,
    'scheduler_gamma': 0.1,
    'model_dir': './model',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f'  Batch {batch_idx+1}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc

def train():
    """Main training function."""
    print('=' * 60)
    print('MNIST Classifier Training')
    print('=' * 60)
    print(f'Configuration: {CONFIG}')
    print()
    
    # Setup
    device = torch.device(CONFIG['device'])
    print(f'Using device: {device}')
    
    # Load data
    train_loader, test_loader, _, _ = download_and_prepare_data()
    
    # Initialize model
    model = MNISTClassifier().to(device)
    print(f'Model parameters: {count_parameters(model):,}')
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = StepLR(
        optimizer,
        step_size=CONFIG['scheduler_step'],
        gamma=CONFIG['scheduler_gamma']
    )
    
    # Training loop
    best_acc = 0.0
    os.makedirs(CONFIG['model_dir'], exist_ok=True)
    
    print('\nStarting training...')
    print('-' * 60)
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        start_time = time.time()
        
        print(f'\nEpoch {epoch}/{CONFIG["epochs"]}')
        print(f'Learning rate: {scheduler.get_last_lr()[0]:.6f}')
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        print(f'  Epoch time: {epoch_time:.2f}s')
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            model_path = os.path.join(CONFIG['model_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss,
            }, model_path)
            print(f'  *** New best model saved! Acc: {test_acc:.2f}% ***')
    
    # Save final model for deployment
    final_path = os.path.join(CONFIG['model_dir'], 'mnist_classifier.pth')
    torch.save(model.state_dict(), final_path)
    
    print('\n' + '=' * 60)
    print('Training Complete!')
    print(f'Best Test Accuracy: {best_acc:.2f}%')
    print(f'Model saved to: {final_path}')
    print('=' * 60)
    
    return model, best_acc

if __name__ == '__main__':
    train()
