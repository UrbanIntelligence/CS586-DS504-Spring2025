import os
import numpy as np

# IMPORTANT: Use a non-interactive backend for matplotlib in headless mode
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.insert(0, "/autograder/submission")

from training import Generator

# Define the device for the training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(model_path, z_dim):
    '''
    Load the saved generator model from the specified path.
    '''
    try:
        model = torch.load(model_path, map_location=device)  # Directly load the model
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

def generate_images(model, num_images, z_dim):
    '''
    Take the model as input and generate a specified number of images.
    '''
    images = []
    try:
        for _ in range(num_images):
            z = torch.randn(1, z_dim, device=device)
            generated_image = model(z).detach().cpu().numpy().reshape(28, 28)
            images.append(generated_image)
    except Exception as e:
        raise RuntimeError(f"Error generating images: {e}")

    return images

def plot_images(images, grid_size, output_path):
    try:
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10), tight_layout=True)
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(images[i], cmap='gray')
            ax.axis("off")
        plt.savefig(output_path)  # Save the plot to the specified path
        plt.close(fig)  # Close the plot to free memory
    except Exception as e:
        raise RuntimeError(f"Error plotting images: {e}")

if __name__ == "__main__":
    import argparse
    import json

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate and evaluate images.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the student-submitted model.")
    parser.add_argument("--num_images", type=int, default=25, help="Number of images to generate.")
    parser.add_argument("--output", type=str, default="/autograder/results/generated_images.png", 
                        help="Output file for generated images.")
    parser.add_argument("--z_dim", type=int, default=100, help="Dimension of the latent space for the Generator.")

    args = parser.parse_args()

    # We'll assign 20 points if everything succeeds, otherwise 0
    MAX_SCORE = 20

    try:
        # 1. Load the model
        model = load_model(args.model_path, args.z_dim)

        # 2. Generate images
        images = generate_images(model, args.num_images, args.z_dim)

        # 3. Plot and save the images
        grid_size = int(np.sqrt(args.num_images))
        if grid_size ** 2 != args.num_images:
            raise ValueError("Number of images must be a perfect square for grid plotting.")

        os.makedirs("/autograder/results", exist_ok=True)
        plot_images(images, grid_size, args.output)

        # If we got here, everything worked fine
        message = f"Images saved to {args.output}"

        # Build a successful JSON structure
        results = {
            "score": MAX_SCORE,            # Full score = 20
            "output": message,            
            "visibility": "visible", 
            "tests": [
                {
                    "name": "Model Generation Test",
                    "score": MAX_SCORE,
                    "max_score": MAX_SCORE,
                    "output": message
                }
            ]
        }

    except Exception as e:
        # On error, assign 0 points and include the error message
        error_message = f"Error during evaluation: {e}"
        print(error_message)

        results = {
            "score": 0,
            "output": error_message,
            "visibility": "visible",
            "tests": [
                {
                    "name": "Model Generation Test",
                    "score": 0,
                    "max_score": MAX_SCORE,
                    "output": error_message
                }
            ]
        }

    # Finally, write results.json to /autograder/results/
    with open("/autograder/results/results.json", "w") as f:
        json.dump(results, f)
    print("Autograder finished.")
