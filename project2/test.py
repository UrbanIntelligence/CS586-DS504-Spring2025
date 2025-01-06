# test_model.py

import torch
from torch.utils.data import DataLoader, Dataset
from model import TaxiDriverClassifier, TaxiDriverClassifier2
from extract_feature import load_data, preprocess_data

from train import TaxiDriverDataset

# Assuming the extract_feature.py handles preprocessing in a generalized way that can be applied to any data

        
def test(model, test_loader, device):
    """
    Test the model performance on the test set.
    """
    model.eval()
    test_loss = 0
    test_correct = 0
    criterion = torch.nn.CrossEntropyLoss()

    ###########################
    # YOUR IMPLEMENTATION HERE #

    ###########################

    return test_loss, test_correct


def test_model():
    """
    Main function to initiate the model testing process.
    Includes loading test data, setting up the model and test loader,
    and executing the testing function.
    """
    # Get the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ###########################
    # YOUR IMPLEMENTATION HERE #

    ###########################