import torch 
import torch.nn as nn

from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple, Literal
from omegaconf import DictConfig
from hydra.utils import instantiate

from ..utils.loss import temporal_loss, spectral_loss
from ..utils.transforms import SpectogramTransform, InverseSpectogramTransform
from ..utils.parent_lightning_pretrainer import FoundationalModelOutput, Pretraining_Lightning, LightningConfig


@dataclass
class ConvBlockConfig:

    in_channels: int 
    out_channel_ratio: float
    shortcut: bool
    n_conv: int = 2
    batch_norm: bool = True
    out_channels: Optional[int] = None
    activation: DictConfig = field(default_factory=lambda: {"_target_": "torch.nn.ReLU"})


class ConvBlock(nn.Module):
    def __init__(self, config: ConvBlockConfig):
        super(ConvBlock, self).__init__()

        self.in_channels = config.in_channels
        self.out_channels = int(self.in_channels * config.out_channel_ratio) if config.out_channels is None else config.out_channels
        self.shortcut = config.shortcut
        self.n_conv = config.n_conv

        conv_layers = [] 

        for i in range(self.n_conv):
            out_channels = self.out_channels if i == self.n_conv - 1 else self.in_channels
            conv_layers.append(
                nn.Conv1d(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )   

            if config.batch_norm:
                conv_layers.append(nn.BatchNorm1d(out_channels))
            
            if i < self.n_conv - 1:
                conv_layers.append(instantiate(config.activation))

        
        #make the shortcut if needed

        if self.shortcut:
            self.shortcut = nn.Linear(self.in_channels, self.out_channels) if self.in_channels != self.out_channels else nn.Identity()
        
        self.conv_layers = nn.Sequential(*conv_layers)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("x.shape", x.shape)
        out = self.conv_layers(x.view([-1] + list(x.shape[-2:])))
        out = out.view(x.shape[:-2] + out.shape[-2:])
        # print("out.shape", out.shape)
        if self.shortcut:
            out += self.shortcut(x.transpose(-1,-2)).transpose(-1,-2)
        
        return out
    

@dataclass
class SingleChannelConvCNNConfig:

    in_channels: int
    reduction_ratios: List[int]
    shortcut: bool
    n_conv: int = 2
    batch_norm: bool = True
    target_out_channels: Optional[int] = None
    activation: DictConfig = field(default_factory=lambda: {"_target_": "torch.nn.ReLU"})
            
        
class SingleChannelConvCNN(nn.Module):

    def __init__(self, config: SingleChannelConvCNNConfig):
        super(SingleChannelConvCNN, self).__init__()

        self.in_channels = config.in_channels
        self.reduction_ratios = config.reduction_ratios
        self.shortcut = config.shortcut
        self.n_conv = config.n_conv
        self.batch_norm = config.batch_norm
        self.activation = config.activation

        self.conv_blocks = nn.ModuleList()
        self.out_channels = config.in_channels
        for i, reduction_ratio in enumerate(self.reduction_ratios):
            conv_block_config = ConvBlockConfig(
                in_channels=self.out_channels,
                out_channel_ratio=reduction_ratio,
                shortcut=self.shortcut,
                n_conv=self.n_conv,
                batch_norm=self.batch_norm,
                activation=self.activation,
                out_channels=config.target_out_channels if i == len(self.reduction_ratios) - 1 else None
            )

            self.conv_blocks.append(ConvBlock(conv_block_config))
            self.out_channels = int(self.out_channels * reduction_ratio)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        return x
    
@dataclass
class SingleChannelAE_CNN_Config:

    #spectogram parameters
    n_fft: int
    hop_length: int
    log_spectogram: bool
    predict_phases: bool


    reduction_ratios: List[int]
    shortcut: bool
    n_conv: int = 2
    batch_norm: bool = True
    activation: DictConfig = field(default_factory=lambda: {"_target_": "torch.nn.ReLU"})


    losses:List[Literal["spectral", "temporal"]] = field(default_factory=lambda:  ["spectral", "temporal"])
    loss_weights:Union[List[float], Literal["equal","balancing_0th_order"]] = "equal", #ToDO: allow for balancing_0th_order and potentially balancing_1st_order
    P:Union[float, List[float]] = 2.0

    def __post_init__(self):
        if isinstance(self.losses, str):
            if self.loss_weights == "equal":
                self.loss_weights = [1/len(self.losses)] * len(self.losses)
            else:
                raise NotImplementedError("Only equal weights are supported for now")
        else:
            #check its the same size
            assert len(self.losses) == len(self.loss_weights)

            #renormalize
            self.loss_weights = [w/sum(self.loss_weights) for w in self.loss_weights]

        if isinstance(self.P, float):
            self.P = [self.P] * len(self.losses)
        else:
            assert len(self.P) == len(self.losses)


@dataclass
class SingleChannelAE_CNN_Output(FoundationalModelOutput):
    output_specs: torch.Tensor

class SingleChannelAE_CNN(nn.Module):
    INPUT_NUM_CHANNELS = 1
    encoder_only_mode = False
    def __init__(self, config: SingleChannelAE_CNN_Config):
        super(SingleChannelAE_CNN, self).__init__()
        
        self.config = config    

        self.forward_transform = SpectogramTransform(
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            reshape_size=-1,
            predict_phases=config.predict_phases,
            log = config.log_spectogram
        )

        self.inverse_transform = InverseSpectogramTransform(
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            original_size=-1,
            predict_phases=config.predict_phases,
            log = config.log_spectogram,
        )

        n_channels = (config.n_fft//2 + 1) * ( 1 + config.predict_phases)

        self.predict_phases = config.predict_phases

        self.losses = config.losses
        self.loss_weights = config.loss_weights
        self.P = config.P

        self.encoder = SingleChannelConvCNN(
            SingleChannelConvCNNConfig(
                in_channels=n_channels,
                reduction_ratios=config.reduction_ratios,
                shortcut=config.shortcut,
                n_conv=config.n_conv,
                batch_norm=config.batch_norm,
                activation=config.activation
            )
        )
        print("encoder out_channels", self.encoder.out_channels)

        self.decoder = SingleChannelConvCNN(
            SingleChannelConvCNNConfig(
                in_channels=self.encoder.out_channels,
                reduction_ratios=[1/s for s in config.reduction_ratios[::-1]],
                shortcut=config.shortcut,
                n_conv=config.n_conv,
                batch_norm=config.batch_norm,
                activation=config.activation,
                target_out_channels=n_channels
            )
        )


    def norm_spectogram(self, x: torch.FloatTensor) -> torch.FloatTensor:
        mean = x.mean(dim=(-1,-2), keepdim=True)
        var = x.var(dim=(-1,-2), keepdim=True)

        return (x - mean) / (var + 1.0e-6) ** 0.5

    def denorm_spectogram(self, predicted_spec: torch.FloatTensor, original_spec: torch.FloatTensor) -> torch.FloatTensor:
        mean = original_spec.mean(dim=(-1,-2), keepdim=True)
        var = original_spec.var(dim=(-1,-2), keepdim=True)

        return predicted_spec * (var + 1.0e-6) ** 0.5 + mean

    def spectral_loss(self, reconstructed_spec: torch.FloatTensor,
                      original_spec: torch.FloatTensor,
                      original_waveform: torch.FloatTensor) -> torch.Tensor:
        # print(reconstructed_spec)
        # print("reconstructed_spec.shape", reconstructed_spec.shape)
        # print("original_spec.shape", original_spec.shape)
        # print(torch.mean(torch.abs(self.denorm_spectogram(reconstructed_spec, original_spec))))
        # print(original_spec)
        # raise ValueError("stop here")
        return spectral_loss(reconstructed_spec,
                                self.norm_spectogram(original_spec))
    
    def temporal_loss(self, reconstructed_spec: torch.FloatTensor,
                      original_spec: torch.FloatTensor,
                      original_waveform: torch.FloatTensor,
                      phases: Optional[torch.FloatTensor] = None,) -> torch.Tensor:

        reconstructed_waveform = self.inverse_transform(self.denorm_spectogram(reconstructed_spec, original_spec),
                                                            phases)
        
        return temporal_loss(reconstructed_waveform, original_waveform)
    

    def forward_loss(self, reconstructed_spec:torch.FloatTensor,
                     input_image: torch.FloatTensor,
                     input_sequence: torch.FloatTensor,
                     phases: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        
        """calculate the losses"""

        loss = 0
        loss_dict = {}
        for i,loss_name in enumerate(self.losses):
            weight = self.loss_weights[i]
            p = self.P[i]
            if loss_name == "spectral":
                l = self.spectral_loss(reconstructed_spec, input_image, input_sequence)
            elif loss_name == "temporal":
                l = self.temporal_loss(reconstructed_spec, input_image, input_sequence, phases)
            else:
                raise ValueError(f"Unknown loss {loss_name}")
            if weight > 0:
                loss += weight * l #allows us to use some losses for logging only
            loss_dict[loss_name] = (l.item(), weight)
        
        # if self.loss_weights_type != "constant":
        #     raise NotImplementedError("Balancing methods not implemented yet")
        
        return loss, loss_dict

    def forward(self, input_waveforms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
         
        if self.predict_phases:
            input_specs = self.forward_transform(input_waveforms)
            phases = None
        else:
            input_specs,phases = self.forward_transform(input_waveforms)

        input_specs_normalized = self.norm_spectogram(input_specs)
        # print("input_specs_normalized.shape", input_specs_normalized.shape)
        latent = self.encoder(input_specs_normalized)
        if self.encoder_only_mode:
            # print("latent.shape", latent.shape)
            return latent
        out = self.decoder(latent)

        loss, loss_dict = self.forward_loss(out, input_specs, input_waveforms, phases)
        # raise NotImplementedError("Need to implement the output of the model")
        return SingleChannelAE_CNN_Output(
            loss=loss,
            losses=loss_dict,
            output_specs = out,
            phases = phases,
            input_specs = input_specs,
            input_waveforms = input_waveforms
        )
    
    def reconstruct_spectograms(self, model_output: SingleChannelAE_CNN_Output) -> List[Tuple[torch.FloatTensor, str]]:
        
        input_specs = model_output.input_specs
        reconstructed_specs = self.denorm_spectogram(model_output.output_specs, input_specs).detach().cpu()

        input_specs = input_specs.detach().cpu()

        channel_names = ["mag"] if not self.predict_phases else ["mag", "phase"]

        output = []

        #for each batch
        for i in range(input_specs.shape[0]):
            output_i = []
            names_i = []
            for j in range(input_specs.shape[1]):
                #for each channel
                #add 2 things, the input spectogram, and the reconstructed spectogram
                output_i += [input_specs[i,j], reconstructed_specs[i,j]]
                prefix = f"channel_{j//len(channel_names)}_{channel_names[j%len(channel_names)]}"
                names_i += [f"{prefix}_input", f"{prefix}_reconstructed"]
            output.append((output_i, names_i))
        
        return output
    
    def reconstruct_waveforms(self, model_output: SingleChannelAE_CNN_Output) -> List[Tuple[torch.FloatTensor, str]]:

        input_waveforms = model_output.input_waveforms
        reconstructed_specs = self.denorm_spectogram(model_output.output_specs, model_output.input_specs)
        reconstructed_waveforms = self.inverse_transform(reconstructed_specs, phases = model_output.phases)

        #convert the spectogram to the waveform
        reconstructed_waveforms = self.inverse_transform(reconstructed_specs, phases = model_output.phases) #shape of (batch_size, n_channels, sequence_len)

        output = []
        #similar to the spectograms
        for i in range(input_waveforms.shape[0]):
            output_i = []
            names_i = []
            for j in range(input_waveforms.shape[1]):
                output_i.append(input_waveforms[i,j].cpu())
                output_i.append(reconstructed_waveforms[i,j].detach().cpu())

                names_i.append(f"channel_{j}_input")
                names_i.append(f"channel_{j}_reconstructed")
            
            output.append((output_i, names_i))
        
        return output
    
    def encoder_only(self, freeze: bool = True):

        self.encoder_only_mode = True

        del self.decoder

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
    def get_encoder_output_size(self):
        return self.encoder.out_channels




class SingleChannelAE_CNN_Lightning(Pretraining_Lightning):
    def __init__(self, LightningConfig: LightningConfig):
                 
        super().__init__(LightningConfig)
        self.model = SingleChannelAE_CNN(LightningConfig.model_config)