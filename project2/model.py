# model.py

import torch.nn as nn

class TaxiDriverClassifier(nn.Module):
    """
    Input:
        Data: the output of process_data function.
        Model: your model.
    Output:
        prediction: the predicted label(plate) of the data, an int value.
    """
    def __init__(self, input_dim, output_dim):
        super(TaxiDriverClassifier, self).__init__()

        ###########################
        # YOUR IMPLEMENTATION HERE #
        ###########################

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        ###########################
        return x
    
