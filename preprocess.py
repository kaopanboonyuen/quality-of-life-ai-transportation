"""
preprocess.py

Description:
This script handles data preprocessing for the Transportation Mobility Factor Extraction project.
It prepares the dataset for training, evaluation, and inference by performing necessary transformations.

Author:
Teerapong Panboonyuen (Kao Panboonyuen)

Usage:
python preprocess.py --data_path /path/to/data --output_path /path/to/output

Dependencies:
- numpy
- pandas
- opencv-python
- Pillow
"""

import os
import argparse
import cv2
import numpy as np
from PIL import Image

def preprocess_data(data_path, output_path):
    """Preprocess the dataset."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for filename in os.listdir(data_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(data_path, filename)
            img = cv2.imread(img_path)
            # Perform preprocessing steps (e.g., resizing, normalization)
            img = cv2.resize(img, (256, 256))
            output_file = os.path.join(output_path, filename)
            cv2.imwrite(output_file, img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input data")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save preprocessed data")
    args = parser.parse_args()
    preprocess_data(args.data_path, args.output_path)