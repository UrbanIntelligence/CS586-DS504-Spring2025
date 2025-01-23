# test_model.py

import os
import torch
from torch.utils.data import DataLoader, Dataset
from model import SiameseLSTM

from train import load_data, train_model, evaluate, SiameseDataset

def test_model(test_dir):
    """
    Initiate the model testing process, including:
    - Loading the saved model
    - Loading and preprocessing test data from test_dir
    - Creating a DataLoader for testing
    - Evaluating the model and printing results
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Construct the file pattern using test_dir
    test_file_pattern = os.path.join(test_dir, "*.csv")
    # Load test data
    X_test, y_test = load_data(test_file_pattern)

    # Get the device

    
    ###########################
    # YOUR IMPLEMENTATION HERE #

    ###########################


    # Print the accuracy in the required format
    print(f"Accuracy={test_accu:.4f}")
