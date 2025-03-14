import torch
import torch.nn as nn
from typing import List, Literal, Optional, Dict
from omegaconf import DictConfig
from hydra.utils import instantiate

from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
    TDSConv2dBlock,
    TDSFullyConnectedBlock,
)


class TDSConvBlock(nn.Module):
    def __init__(self, num_features, channels, kernel_width):

        super().__init__()

        self.conv = TDSConv2dBlock(channels, num_features // channels, kernel_width)
        self.fnn = TDSFullyConnectedBlock(num_features)

    def forward(self, x):
        # print("type", type(x))
        x = self.conv(x)
        x = self.fnn(x)
        return (x,) #unified output format

class CNN_RNN_Encoder(nn.Module):
    #striped CNN RNN encoder

    def __init__(self, 
                 num_features: int,
                 stripping:List[Literal["CNN", "RNN"]] = ["CNN","RNN","CNN","RNN"],
                 CNN_config: Optional[DictConfig] = None,
                    RNN_config: Optional[DictConfig] = None) -> None:
        
        super().__init__()
        print("here")
        self.stripping = stripping

        layers = []
        for i, layer in enumerate(stripping):
            if layer == "CNN":
                layers.append(instantiate(CNN_config, num_features = num_features))
            elif layer == "RNN":
                layers.append(instantiate(RNN_config, hidden_size = num_features//2 if RNN_config.bidirectional else num_features, input_size = num_features))
            else:
                raise ValueError(f"Unknown layer type: {layer}")

        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers:
            x,*_ = layer(x)
            # print("x", x.shape)
        return x