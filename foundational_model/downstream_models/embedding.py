import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleEmbedding(nn.Module):
    """Simple embedding layer that takes in a tensor of shape (batch_size, channels, embedding_size)
    and returns a tensor of shape (batch_size, embedding_size)"""

    def __init__(self, embedding_size, n_channels=2):
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros((2, n_channels, embedding_size)))

    def forward(self, x):   
        #x is of shape (batch_size, 2, n_channels, embedding_size)

        return x + self.embedding.unsqueeze(0)
    


