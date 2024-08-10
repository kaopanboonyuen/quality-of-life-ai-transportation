"""
train.py

Description:
This script trains the model using the preprocessed dataset. It includes model initialization, training, and saving.

Author:
Teerapong Panboonyuen (Kao Panboonyuen)

Usage:
python train.py --config config.yaml

Dependencies:
- torch
- torchvision
- numpy
- pyyaml
"""

import os
import argparse
import yaml
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_model(config):
    """Train the model based on the provided configuration."""
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, config['num_classes'])
    
    # Prepare data loaders
    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor()])
    train_dataset = torchvision.datasets.ImageFolder(root=config['train_data'], transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    model.train()
    for epoch in range(config['epochs']):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {loss.item()}")
    
    # Save the model
    torch.save(model.state_dict(), config['model_save_path'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()
    config = load_config(args.config)
    train_model(config)