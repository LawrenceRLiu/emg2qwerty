import torch
import torch.nn as nn
import torch.nn.functional as F


from emg2qwerty.lightning import LightningModule

from .utils import make_mlp
from .pool import MLP_Pool, AttentionPool
from .embedding import SimpleEmbedding


#quick and dirty without hydra rn
class DownstreamDecoder(nn.Module):
    """Decoder for the downstream task"""

    def __init__(self, 
                 input_size:int = 14**2*8,
                 pooling_method:str = "attention",
                 pooling_kwargs:dict = {},
                 embedding_method:str = "simple",
                embedding_kwargs:dict = {},
                n_layers:int = 2,
                reduction_ratio:float = 0.5,
                output_size:int = 26,
                activation:str = 'relu'):

        super().__init__()
        self.input_size = input_size

        self.pooling = self._get_pooling(pooling_method, pooling_kwargs)

        self.embedding = self._get_embedding(embedding_method, embedding_kwargs)

        self.mlp = make_mlp(input_size, output_size,
                                n_layers=n_layers, reduction_ratio=reduction_ratio,
                                activation=activation)
        
        self.final_activation = nn.LogSoftmax(dim=-1)
    

    def _get_pooling(self,pooling_method:str, pooling_kwargs:dict = {})
        
        pooling_method = pooling_method.lower()
        if pooling_method == "mlp":
            self.pool = MLP_Pool(
                embedding_size = self.input_size,
                **pooling_kwargs)
        elif pooling_method == "attention":
            self.pool = AttentionPool(embedding_size = self.input_size,
                **pooling_kwargs)
        else:
            raise ValueError(f"pooling method {pooling_method} not supported")


    def _get_embedding(self, embedding_method:str, embedding_kwargs:dict = {}):
        embedding_method = embedding_method.lower()
        if embedding_method == "simple":
            self.embedding = SimpleEmbedding(
                **embedding_kwargs)
        else:
            raise ValueError(f"embedding method {embedding_method} not supported")
        
    
    def forward(self, x):   
        #x is expected to be of shape (batch_size, 2, channels, embedding_size)

        #add the positional embedding
        x = self.embedding(x)

        #pool across the channels
        x = self.pool(x) #shape (batch_size, embedding_size)

        #pass through the mlp
        x = self.mlp(x)

        #apply the final activation
        x = self.final_activation(x)

        return x
    
