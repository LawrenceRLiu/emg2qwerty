import torch
import torch.nn as nn
import torch.nn.functional as F


def make_mlp(input_size, output_size, 
             n_layers:int = 2, reduction_ratio:float = 0.5,
             activation:str = 'relu'):
    """Creates a MLP with the given parameters"""
    if n_layers == 0:
        return nn.Identity()

    layers = []

    n_in = input_size

    for i in range(n_layers):
        n_out = int(n_in * reduction_ratio)
        if n_out < output_size:
            n_out = output_size
            print(f"warning: n_out is less than the output size {output_size}, setting to {output_size}")
        layers.append(nn.Linear(n_in, n_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'gelu':
            layers.append(nn.GELU())
        else:
            raise ValueError(f"activation {activation} not supported")
        n_in = n_out
    layers.append(nn.Linear(n_in, output_size))
    return nn.Sequential(*layers)