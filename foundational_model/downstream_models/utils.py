import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from hydra.utils import instantiate
from typing import List, Optional, Dict, Union

class MLP(nn.Module):

    def __init__(self, input_size:int, output_size:int,
                 activation:DictConfig = {"_target_":"torch.nn.ReLU"},    
                 hidden_dims:Optional[List[int]] = None,
                 dropout:float = 0.0,
                 n_layers:int = 2, 
                 reduction_ratio:float = 0.5,
                 final_activation:Optional[Union[DictConfig,bool]] = False,
                 final_bias:bool = True):
        """MLP with configurable number of layers and sizes

        Args:
            input_size (int): the size of the input
            output_size (int): the size of the output
            activation (DictConfig): the activation function to use
            hidden_dims (_type_, optional): the sizes of the intermediate layers. Defaults to None. in which case the intermediate sizes are calculated using reduction_ratio
            n_layers (int, optional): the number of layers in the MLP. Defaults to 2.
            reduction_ratio (float, optional): the reduction ratio of the layers. Defaults to 0.5. 
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        if hidden_dims is not None and n_layers != 0:
            n_layers = len(hidden_dims) + 1
        print("n_layers", n_layers)
        if n_layers == 1:
            self.mlp = nn.Linear(input_size, output_size)
            return
        if n_layers < 1:
            self.mlp = nn.Identity()
            self.output_size = input_size
            return

        
        layers:List[nn.Module] = []

        n_in = input_size

        if hidden_dims is None:
            hidden_dims = [int(n_in * (reduction_ratio ** (i+1))) for i in range(n_layers-1)]
            #raise a warning if we are inadvertedly reducing the size to less than the output size
            if hidden_dims[-1] < output_size:
                print(f"warning: n_out is less than the output size {output_size}, setting to {output_size}")
                hidden_dims = [max(output_size, s) for s in hidden_dims]
            hidden_dims.append(output_size)
        
        print(instantiate(activation))
        for i, n_out in enumerate(hidden_dims):
            layers.append(nn.Linear(n_in, n_out, bias=True if i != len(hidden_dims) - 1 else final_bias))
            layers.append(instantiate(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            n_in = n_out
        if isinstance(final_activation, DictConfig):
            layers.append(instantiate(final_activation))
        elif final_activation:
            layers.append(instantiate(activation))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    

