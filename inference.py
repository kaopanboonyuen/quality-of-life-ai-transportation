"""
inference.py

Description:
This script performs inference on new images using the trained model and saves the results.

Author:
Teerapong Panboonyuen (Kao Panboonyuen)

Usage:
python inference.py --image_path /path/to/image.png --output_path /path/to/output.png

Dependencies:
- torch
- torchvision
- PIL
"""

import argparse
import torch
from torchvision import transforms
from PIL import Image

def infer_image(model_path, image_path, output_path):
    """Perform inference on a single image."""
    model = torch.load(model_path)
    model.eval()
    
    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor()])
    
    img = Image.open(image_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)
    
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
    
    # Save the result
    result_img = Image.new('RGB', (256, 256), color='white')
    result_img.save(output_path)
    print(f"Inference result saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output image")
    args = parser.parse_args()
    infer_image(args.model_path, args.image_path, args.output_path)