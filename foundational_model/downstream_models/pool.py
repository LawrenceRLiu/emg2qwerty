import torch
import torch.nn as nn


from omegaconf import DictConfig
from hydra.utils import instantiate
from typing import List, Optional, Dict


from .utils import MLP

#different pooling methods, all have embedding_size as input

class AvgPool(nn.Module):
    """averages across the channels"""

    def __init__(self,
                 embedding_size:int):

        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=1, stride=1)

    def forward(self, x):
        # x is of shape (batch_size, 2, channels, embedding_size)
        # we want to average across the channels and hands
        x = self.pool(x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))
        return x[:, 0, :] #shape (batch_size, embedding_size)
    
class MaxPool(nn.Module):

    """max across the channels"""

    def __init__(self,
                    embedding_size:int):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=1, stride=1)

    def forward(self, x):
        # x is of shape (batch_size, 2, channels, embedding_size)
        # we want to max across the channels and hands
        x = self.pool(x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))
        return x[:, 0, :] #shape (batch_size, embedding_size)
    



class MLP_Pool(nn.Module):
    """MLP to pool across the channels"""

    def __init__(self, embedding_size:int,
                 n_channels:int,
                 MLP_config:DictConfig):
        super().__init__()
        self.pool = instantiate({"__target__":MLP},
                                input_size=embedding_size * n_channels,
                                output_size=embedding_size,
                                **MLP_config)
        self.n_channels = n_channels

    def forward(self, x):
        # x is of shape (batch_size, 2, channels, embedding_size)
        # we want to pool across the channels and hands
        x = self.pool(x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
        return x 
    
class AttentionPool(nn.Module):
    """Uses attention to pool across the channels"""

    def __init__(self, embedding_size:int,
                 n_heads:int,
                 attention_MLP:DictConfig,
                 value_MLP:DictConfig):
        super().__init__()
        assert n_heads> 0, "n_heads must be greater than 0"
        assert embedding_size % n_heads == 0, "embedding_size must be divisible by n_heads"
        self.attention_network = instantiate({"__target__":MLP}, 
                                             input_size=embedding_size,
                                             output_size=n_heads,
                                                **attention_MLP)
        self.MLP = instantiate({"__target__":MLP},
                               input_size=embedding_size,
                              output_size=embedding_size,
                                **value_MLP)
        self.n_heads = n_heads

    def forward(self, x):
        x_use = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3]) #shape (batch, 2*channels, embedding_size)
        #pass through attention network
        attention_logits = self.attention_network( x_use ) #shape (batch, 2*channels, n_heads)
        
        attention_weights = torch.softmax(attention_logits, dim=1) #shape (batch, 2*channels, n_heads)

        emebddings = self.MLP(x) #shape (batch, 2*channels, embedding_size)

        #apply attention weights
        pooled = (
            attention_weights.unsqueeze(-1) * \
            emebddings.reshape(x_use.shape[0], x_use.shape[1], self.n_heads, -1)
            ).sum(dim=1) #shape (batch, n_heads, embedding_size)
        return pooled.reshape(x.shape[0], -1) #shape (batch, n_heads*embedding_size)
    


