import torch
import torch.nn as nn
import torch.nn.functional as F
from emg2qwerty.data import LabelData
from emg2qwerty.decoder import Decoder

from typing import Tuple

class ParentModel(nn.Module):
    decoder:Decoder
    model:nn.Module

    def forward_and_decode(self, input:torch.FloatTensor, input_length:torch.LongTensor) -> Tuple[torch.FloatTensor, LabelData]:
        """Forward pass and decoding of the model output.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, LabelData]: The model output and the decoded label.
        """
        output = self.model(x)
        decoded = self.decoder.decode_batch(
            emissions=output.detach().cpu().numpy(),
            emission_lengths=input_length.detach().cpu().numpy(),
        )
        return output, decoded
    
    