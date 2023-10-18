import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch 

class SimpleSRClassifier(nn.Module):
    """
    this class includes a basic classifier with one hidden layer and one output layer.
    :param input_dim: the dimension of the input vector where only embedding size is needed, batch size is implicit
    :param hidden_dim : the dimension of the hidden layer, batch size is implicit
    :param label_nr : the dimension of the output layer, i.e. the number of labels
    """
    def __init__(self, input_dim, hidden_dim, label_nr, dropout_rate, device='cpu'):

        super(SimpleSRClassifier, self).__init__()
        self.device = device
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, label_nr)
        self.dropout_rate = dropout_rate
        
    def forward(self, batch):
        """
        this function takes one word and applies a non-linear matrix transformation (hidden layer)
        Its output is then fed to an output layer. Then it returns the concatenated and transformed vectors.
        :param batch: the batch of words 
        :return: the transformed vectors after output layer
        """
        words = batch['words']
        if torch.cuda.is_available():
            words = words.to(self.device)

        x = F.relu(self.hidden_layer(words))
        x = F.dropout(x, p=self.dropout_rate)
        return torch.sigmoid(self.output_layer(x).squeeze())



class ContextSRClassifier(nn.Module):

    def __init__():
        pass