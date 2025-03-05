import torchaudio.transforms as T_audio
import torchvision.transforms as T_vision
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union, Optional, Callable, Literal

@dataclass
class SpectogramTransform(nn.Module):
    n_fft: int = 64
    hop_length: int = 16
    reshape_size: int = 224
    predict_phases: bool = False
    eps: float = 1e-6
    log: bool = True

    def __post_init__(self):
        super().__init__()  
        self.spectrogram = T_audio.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=None)
        self.resize = T_vision.Resize((self.reshape_size, self.reshape_size), 
                                      interpolation=T_vision.InterpolationMode.NEAREST)

    def forward(self, x: torch.FloatTensor
                ) -> Union[Tuple[torch.FloatTensor, torch.FloatTensor], torch.FloatTensor]:
        """performs the spectogram on the input waveform

        Args:
            x (torch.FloatTensor): the waveform of shape (..., Seq_length)

        Returns:
            if predict_phases:
                torch.FloatTensor: the magnitude spectrogram of shape (..., 2, reshape_size, reshape_size) where 
                the first channel is the magnitude and the second channel is the phase
            else:
                torch.FloatTensor: the magnitude spectrogram of shape (..., 1, reshape_size, reshape)
                torch.FloatTensor: the phase spectrogram of shape (..., 1, n_fft//2 + 1, sequence_len//hop_length + 1)
        """

        x = self.spectrogram(x) #shape (..., n_fft//2 + 1, sequence_len//hop_length + 1)

        mag = torch.abs(x)
        # #convert to log domain
        if self.log:
            print("taking log")
            mag = torch.log(mag + self.eps)
  
        phase = torch.angle(x)

        #if we are predicting phases, stack and resize
        if self.predict_phases:
            x = torch.stack([mag, phase], dim=-3)
            x = self.resize(x)
            return x
        else:
            mag = self.resize(mag)
            return (mag, phase)

@dataclass
class InverseSpectogramTransform(nn.Module):
    n_fft: int = 64
    hop_length: int = 16
    original_size: Tuple[int, int] = (1000, 224)
    predict_phases: bool = False
    eps: float = 1e-6
    log: bool = True

    def __post_init__(self):
        super().__init__()
        self.istft = T_audio.InverseSpectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length)

        self.resize = T_vision.Resize(self.original_size, 
                                      interpolation=T_vision.InterpolationMode.NEAREST)
        self.tensor = torch.tensor([1.0, 1j])

    def forward(self, x: torch.FloatTensor,
                phases: Optional[torch.FloatTensor] = None
                ) -> torch.FloatTensor:
        """_summary_

        Args:
            x (torch.FloatTensor): either the predicted magnitude spectrogram of shape (..., 1, reshape_size, reshape_size) or the predicted waveform spectogram of shape (..., 2, reshape_size, reshape_size)
            phases (Optional[torch.FloatTensor], optional): the phase spectrogram of shape (..., 1, n_fft//2 + 1, sequence_len//hop_length + 1). Defaults to None, necessary if predict_phases is False.
        Returns:
            torch.FloatTensor: the waveform of shape (..., Seq_length)
        """

        #resize
        #flatten the first and second dimensions and the resize
        x = self.resize(x.reshape(-1, *x.shape[-2:])).reshape(*x.shape[:-2], *self.original_size)
        if self.predict_phases:
            if self.log:
                x = torch.exp(torch.einsum("...ijk,i -> ...jk", x, self.tensor)) #multiply by 1+1j to get the complex number
            else:
                x = torch.exp(x[..., 1, :, :] * 1j) * x[..., 0, :, :]
        else:
            if self.log:
                x = torch.exp(1j * phases+x)
            else:
                x = x * torch.exp(1j * phases)
        print("reconstructed spectogram", x)
        assert torch.all(torch.isfinite(x)) 
        x = self.istft(x)
        print(x)
        return x
    