import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from hydra.utils import instantiate
from typing import List, Optional, Dict

class MLP(nn.Module):

    def __init__(self, input_size:int, output_size:int,
                 activation:DictConfig,
                 intermediate_sizes:Optional[List[int]] = None,
                 n_layers:int = 2, 
                 reduction_ratio:float = 0.5):
        """MLP with configurable number of layers and sizes

        Args:
            input_size (int): the size of the input
            output_size (int): the size of the output
            activation (DictConfig): the activation function to use
            intermediate_sizes (_type_, optional): the sizes of the intermediate layers. Defaults to None. in which case the intermediate sizes are calculated using reduction_ratio
            n_layers (int, optional): the number of layers in the MLP. Defaults to 2.
            reduction_ratio (float, optional): the reduction ratio of the layers. Defaults to 0.5. 
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        if intermediate_sizes is not None and n_layers != 0:
            n_layers = len(intermediate_sizes) + 1

        if n_layers == 1:
            self.mlp = nn.Linear(input_size, output_size)
            return
        if n_layers < 1:
            self.mlp = nn.Identity()
            return

        
        layers:List[nn.Module] = []

        n_in = input_size

        if intermediate_sizes is None:
            intermediate_sizes = [int(n_in * (reduction_ratio ** (i+1))) for i in range(n_layers-1)]
            #raise a warning if we are inadvertedly reducing the size to less than the output size
            if intermediate_sizes[-1] < output_size:
                print(f"warning: n_out is less than the output size {output_size}, setting to {output_size}")
                intermediate_sizes = [max(output_size, s) for s in intermediate_sizes]
            intermediate_sizes.append(output_size)
        
        for i, n_out in enumerate(intermediate_sizes):
            layers.append(nn.Linear(n_in, n_out))
            layers.append(instantiate(activation))
            n_in = n_out
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    

