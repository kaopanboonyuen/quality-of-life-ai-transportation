"""
evaluate.py

Description:
This script evaluates the trained model using the test dataset and calculates performance metrics.

Author:
Teerapong Panboonyuen (Kao Panboonyuen)

Usage:
python evaluate.py --model_path /path/to/model --test_data /path/to/test_data

Dependencies:
- torch
- torchvision
- numpy
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(model_path, test_data):
    """Evaluate the model and print metrics."""
    model = torch.load(model_path)
    model.eval()
    
    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor()])
    test_dataset = ImageFolder(root=test_data, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())
    
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--test_data", type=str, required=True, help="Path to the test data")
    args = parser.parse_args()
    evaluate_model(args.model_path, args.test_data)