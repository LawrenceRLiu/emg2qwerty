from transformers import ViTMAEConfig, ViTMAEModel, ViTMAEForPreTraining

from transformers.models.vit_mae.modeling_vit_mae import ViTMAEForPreTrainingOutput
import torchaudio.transforms as T_audio
import torchvision.transforms as T_vision
import torch
import torch.nn as nn
from dataclasses import dataclass

from typing import Any, Dict, List, Tuple, Union, Optional, Callable, Literal

#call the model ViTMAEForEMG


class ViTMAEForEMGConfig(ViTMAEConfig):
    def __init__(self, 
        #spectogram configs
        sequence_len:int=1000,
        n_fft:int = 128,
        hop_length:int = 64,
        predict_phases:bool = False, #predict the phases or just pass them along
        losses:List[Literal["temporal_all", "temporal_masked_only", "spectral_masked_only", "spectral_all"]] = ["spectral_masked_only"],
        loss_weights:Union[List[float], Literal["equal","balancing_0th_order"]] = "equal",

        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        patch_size=16,
        qkv_bias=True,
        decoder_num_attention_heads=16,
        decoder_hidden_size=512,
        decoder_num_hidden_layers=8,
        decoder_intermediate_size=2048,
        mask_ratio=0.75,
        norm_pix_loss=False,
        **kwargs):

        print(sequence_len)  
        self.spectogram_size = (n_fft//2 + 1, sequence_len//hop_length + 1)
        print(self.spectogram_size)
        self.reshape_size = max(self.spectogram_size)
        #increase until its a multiple of patch_size
        while self.reshape_size % patch_size != 0:
            self.reshape_size += 1
        print(self.reshape_size)
        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            image_size=self.reshape_size,
            patch_size=patch_size,
            num_channels= 1 if not predict_phases else 2,
            qkv_bias=qkv_bias,
            decoder_num_attention_heads=decoder_num_attention_heads,
            decoder_hidden_size=decoder_hidden_size,
            decoder_num_hidden_layers=decoder_num_hidden_layers,
            decoder_intermediate_size=decoder_intermediate_size,
            mask_ratio=mask_ratio,
            norm_pix_loss=norm_pix_loss,
            **kwargs
        )

        self.losses = losses
        self.loss_weights = loss_weights
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.predict_phases = predict_phases
        self.sequence_len = sequence_len




class SpectogramTransform(nn.Module):

    def __init__(self,
                 n_fft: int = 64,
                 hop_length: int = 16,
                 reshape_size: int = 224,
                 predict_phases: bool = False,
                 eps: float = 1e-6):
        
        super().__init__()

        self.spectrogram = T_audio.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power = None)

        self.eps = eps
        self.predict_phases = predict_phases
        self.resize = T_vision.Resize((reshape_size, reshape_size))

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
        #convert to log domain
        mag = torch.log(mag + self.eps)

        phase = torch.angle(x)

        #if we are predicting phases, stack and resize
        if self.predict_phases:
            x = torch.stack([mag, phase], dim=-3)
            x = self.resize(x)
            return x
        else:
            mag = self.resize(mag)
            return (mag.unsqueeze(-3), phase)
        
class InverseSpectogramTransform(nn.Module):
    
    def __init__(self,
                 n_fft: int = 64,
                 hop_length: int = 16,
                 original_size: Tuple[int, int] = (1000, 224),
                 predict_phases: bool = False,
                 eps: float = 1e-6):
        
        super().__init__()

        self.istft = T_audio.InverseSpectrogram(
            n_fft=n_fft,
            hop_length=hop_length)

        self.eps = eps

        self.resize = T_vision.Resize(original_size)
        self.predict_phases = predict_phases
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
        x = self.resize(x)
        if self.predict_phases:
            x = torch.exp(torch.einsum("...ijk,i -> ...jk", x, self.tensor))
        else:
            print(x.shape)
            print(x.squeeze(-3).shape)
            print(phases.shape)
            x = torch.exp(x.squeeze(-3) + 1j * phases)
        x = self.istft(x)
        return x
    


@dataclass
class ViTMAEForEMG_PretrainingOutput(ViTMAEForPreTrainingOutput):
    """
    Class for ViTMAEForPreTraining's outputs, with potential hidden states and attentions.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`):
            Pixel reconstruction loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        loss_dict (`Dict[str, tuple[torch.FloatTensor,float]`, *optional*, 
            the losses used to calculate the total loss. The key is the loss name and the value is a tuple of the loss value and the weight of the loss.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss_dict: Dict[str, Tuple[torch.FloatTensor, float]] = None

class ViTMAEForEMG_Pretraining(ViTMAEForPreTraining):


    def __init__(self, config: ViTMAEForEMGConfig):
        super().__init__(config)

        self.forward_transform = SpectogramTransform(
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            reshape_size=config.reshape_size,
            predict_phases=config.predict_phases
        )

        self.inverse_transform = InverseSpectogramTransform(
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            original_size=config.spectogram_size,
            predict_phases=config.predict_phases
        )
        
        self.predict_phases = config.predict_phases
        print(config.losses)
        self.losses = config.losses


        if isinstance(config.loss_weights, list):
            self.loss_weights = {loss: weight for loss, weight in zip(config.losses, config.loss_weights)}
            self.loss_weights_type = "constant"
        elif config.loss_weights == "equal":
            self.loss_weights = {loss: 1/len(config.losses) for loss in config.losses}
            self.loss_weights_type = "constant"
        else:
            #todo implement balancing_0th_order
            raise ValueError(f"Unknown loss weight configuration {config.loss_weights}")
    

        self.sequence_len = config.sequence_len


    def temporal_loss_all(self, predicted_patches: torch.FloatTensor,
                            mask: torch.FloatTensor,
                            input_image: torch.FloatTensor,
                            input_sequence: torch.FloatTensor,
                            phases: Optional[torch.FloatTensor] = None,
                            interpolate_pos_encoding: Optional[bool] = False,
                             denorm:bool = True #denorm is needed
                             ) -> torch.FloatTensor:
        """temporal loss function

        Args:
            predicted_patches (torch.FloatTensor): shape of (batch_size, num_patches, patch_size**2 * num_channels)
            target_sequence (torch.FloatTensor): shape of (.., sequence_len)
            phases (Optional[torch.FloatTensor], optional): the phase spectrogram of shape (..., 1, n_fft//2 + 1, sequence_len//hop_length + 1). Defaults to None, necessary if predict_phases is False.

        Returns:
            torch.FloatTensor: MSE loss between the predicted waveform and the target waveform
        """

        if self.config.norm_pix_loss and denorm:
            input_patches = self.patchify(input_image, interpolate_pos_encoding=interpolate_pos_encoding)
            #denorm the predicted patches
            mean = input_patches.mean(dim=-1, keepdim=True)
            var = input_patches.var(dim=-1, keepdim=True)
            predicted_patches = predicted_patches * (var + 1.0e-6) ** 0.5 + mean

        #convert the patches to the original image size
        predicted_spectogram = self.unpatchify(predicted_patches)
        #convert the predicted spectogram to the waveform
        predicted_waveform = self.inverse_transform(predicted_spectogram, phases)
        #calculate the loss
        loss = torch.mean((predicted_waveform - 
                           input_sequence[..., :predicted_waveform.shape[-1]])**2) #cut the target sequence to match the predicted waveform
        return loss
    
    def temporal_loss_masked(self, predicted_patches: torch.FloatTensor,
                            mask: torch.FloatTensor,
                            input_image: torch.FloatTensor,
                            input_sequence: torch.FloatTensor,
                            phases: Optional[torch.FloatTensor] = None,
                            interpolate_pos_encoding: Optional[bool] = False) -> torch.FloatTensor:
        """temporal loss function

        Args:
            predicted_patches (torch.FloatTensor): shape of (batch_size, num_patches, patch_size**2 * num_channels)
            mask (torch.FloatTensor): shape of (batch_size, num_patches), Tensor indicating which patches are masked (1) and which are not (0).
            input_image (torch.FloatTensor): shape of (batch_size, num_channels, height, width)
            input_sequence (torch.FloatTensor): shape of (.., sequence_len)
            phases (Optional[torch.FloatTensor], optional): the phase spectrogram of shape (..., 1, n_fft//2 + 1, sequence_len//hop_length + 1). 
                    Defaults to None, necessary if predict_phases is False.
            interpolate_pos_encoding (bool, optional): whether to interpolate the positional encoding. Defaults to False.

        Returns:
            torch.FloatTensor: MSE loss between the predicted waveform and the target waveform
        """

        input_patches = self.patchify(input_image, interpolate_pos_encoding=interpolate_pos_encoding)

        if self.config.norm_pix_loss:
            #denorm the predicted patches
            mean = input_patches.mean(dim=-1, keepdim=True)
            var = input_patches.var(dim=-1, keepdim=True)
            predicted_patches = predicted_patches * (var + 1.0e-6) ** 0.5 + mean
        
        patches = predicted_patches * mask.unsqueeze(-1) + input_patches * (1 - mask).unsqueeze(-1)
        
        return self.temporal_loss_all(patches, input_sequence, phases, denorm = False)
    
    def spectral_loss_all(self, predicted_patches: torch.FloatTensor, input_image:torch.FloatTensor,
                          interpolate_pos_encoding: Optional[bool] = False) -> torch.FloatTensor:
        """spectral loss for all patches masked or not

        Args:
            predicted_patches (torch.FloatTensor): shape of (batch_size, num_patches, patch_size**2 * num_channels)
            input_image (torch.FloatTensor): shape of (batch_size, num_channels, height, width)
            interpolate_pos_encoding (bool, optional): whether to interpolate the positional encoding. Defaults to False.
        
        Returns:
            torch.FloatTensor: MSE loss between the predicted patches and the target patches
        """

        input_patches = self.patchify(input_image, interpolate_pos_encoding=interpolate_pos_encoding)

        if self.config.norm_pix_loss:
            mean = input_patches.mean(dim=-1, keepdim=True)
            var = input_patches.var(dim=-1, keepdim=True)
            input_patches = (input_patches - mean) / (var + 1.0e-6) ** 0.5

        return torch.mean((predicted_patches - input_patches)**2)
    
    def spectral_loss_masked(self, predicted_patches: torch.FloatTensor,
                             mask: torch.FloatTensor,
                             input_image: torch.FloatTensor,    
                                interpolate_pos_encoding: Optional[bool] = False) -> torch.FloatTensor:
        
        """spectral loss for masked patches

        Args:
            predicted_patches (torch.FloatTensor): shape of (batch_size, num_patches, patch_size**2 * num_channels)
            mask (torch.FloatTensor): shape of (batch_size, num_patches), Tensor indicating which patches are masked (1) and which are not (0).
            input_image (torch.FloatTensor): shape of (batch_size, num_channels, height, width)
            interpolate_pos_encoding (bool, optional): whether to interpolate the positional encoding. Defaults to False.
        
        Returns:
            torch.FloatTensor: MSE loss between the predicted patches and the target patches
        """

        return super().forward_loss(input_image, predicted_patches, mask, interpolate_pos_encoding=interpolate_pos_encoding)

    
    def forward_loss(self, predicted_patches:torch.FloatTensor,
                     mask: torch.FloatTensor,   
                     input_image: torch.FloatTensor,
                     input_sequence: torch.FloatTensor,
                     phases: Optional[torch.FloatTensor] = None,
                     interpolate_pos_encoding: Optional[bool] = False) -> torch.FloatTensor:
        
        """calculate the losses"""

        loss = 0
        loss_dict = {}
        for loss_name in self.losses:
            weight = self.loss_weights[loss_name]
            if loss_name == "temporal_all":
                l = self.temporal_loss_all(predicted_patches, mask, input_image, input_sequence, phases, interpolate_pos_encoding) 
            elif loss_name == "temporal_masked_only":
                l = self.temporal_loss_masked(predicted_patches, mask, input_image, input_sequence, phases, interpolate_pos_encoding) 
            elif loss_name == "spectral_masked_only":
                l = self.spectral_loss_masked(predicted_patches, mask, input_image, interpolate_pos_encoding)
            elif loss_name == "spectral_all":
                l = self.spectral_loss_all(predicted_patches, input_image, interpolate_pos_encoding)
            else:
                raise ValueError(f"Unknown loss {loss_name}")
            
            loss += weight * l
            loss_dict[loss_name] = (l, weight)
        
        if self.loss_weights_type != "constant":
            raise NotImplementedError("Balancing methods not implemented yet")
        
        return loss, loss_dict
        

    def forward(
        self,
        input_emg: torch.FloatTensor,
        noise: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Union[Tuple, ViTMAEForPreTrainingOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.predict_phases:
            pixel_values = self.forward_transform(input_emg)
            phases = None
        else:
            
            pixel_values,phases = self.forward_transform(input_emg)

        print(pixel_values.shape)

        outputs = self.vit(
            pixel_values,
            noise=noise,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        decoder_outputs = self.decoder(latent, ids_restore, interpolate_pos_encoding=interpolate_pos_encoding)
        logits = decoder_outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)

        loss, loss_dict = self.forward_loss(logits, mask, pixel_values, input_emg, phases, interpolate_pos_encoding)

        if not return_dict:
            output = ((logits, mask, ids_restore) + outputs[2:] + (loss_dict)) if loss is not None else (logits, mask, ids_restore) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ViTMAEForEMG_PretrainingOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if __name__ == "__main__":

    example_emg = torch.randn(1, 10000)

    model = ViTMAEForEMG_Pretraining(ViTMAEForEMGConfig(sequence_len = example_emg.shape[-1],
                                                        losses = ["temporal_masked_only","spectral_masked_only"],
                                                        loss_weights = "equal"))

    output = model(example_emg)

    print(output.loss)
    print(output.loss_dict) 

