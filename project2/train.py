# train.py

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from model import TaxiDriverClassifier
from extract_feature import load_data, preprocess_data

if torch.cuda.is_available():
  device = torch.device("cuda:0")
  print("GPU")
else:
  device = torch.device("cpu")
  print("CPU")

class TaxiDriverDataset(Dataset):
    """
    Custom dataset class for Taxi Driver Classification.
    Handles loading and preparing data for the model
    """
    def __init__(self, X, y, device):
        ###########################
        # YOUR IMPLEMENTATION HERE #

        ###########################

def train(model, optimizer, criterion, train_loader, device):
    """
    Function to handle the training of the model.
    Iterates over the training dataset and updates model parameters.
    """
    ###########################
    # YOUR IMPLEMENTATION HERE #

    ###########################
    return train_loss, train_acc

# Define the testing function
def evaluate(model, criterion, test_loader, device):
    """
    Function to evaluate the model performance on the validation set.
    Computes loss and accuracy without updating model parameters.
    """
    ###########################
    # YOUR IMPLEMENTATION HERE #

    ###########################
    return test_loss, test_acc

def train_model():
    """
    Main function to initiate the model training process.
    Includes loading data, setting up the model, optimizer, and criterion,
    and executing the training and validation loops.
    """

    ###########################
    # YOUR IMPLEMENTATION HERE #

    ###########################