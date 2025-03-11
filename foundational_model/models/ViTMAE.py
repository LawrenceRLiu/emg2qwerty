from transformers import ViTMAEConfig, ViTMAEModel, ViTMAEForPreTraining, ViTMAEPreTrainedModel

from transformers.models.vit_mae.modeling_vit_mae import ViTMAEForPreTrainingOutput, ViTMAEDecoder

import torch
import torch.nn as nn
import torchvision.transforms as T_vision
from dataclasses import dataclass, InitVar
import pytorch_lightning as pl
from typing import Any, Dict, List, Tuple, Union, Optional, Callable, Literal

from ..utils.loss import temporal_loss, spectral_loss
from ..utils.transforms import SpectogramTransform, InverseSpectogramTransform
from ..utils.custom_logger import CustomPretrainLogger
from ..utils.parent_lightning_pretrainer import LightningConfig, FoundationalModelOutput, ParentModel, Pretraining_Lightning
# from .output import AdditionalFoundationalModelOutput
#call the model ViTMAEForEMG


class ViTMAEForEMGConfig(ViTMAEConfig):
    def __init__(self, 
        #spectogram configs
        sequence_len:int=1000,
        n_fft:int = 128,
        hop_length:int = 64,
        reshape_size:int = 128,
        interpolation:Union[Tuple[Literal["linear", "nearest", "cubic"]],Literal["linear", "nearest", "cubic"]] = "linear", #supports different resampling methods for transform and inverse transform
        predict_phases:bool = False, #predict the phases or just pass them along
        losses:List[Literal["temporal_all", "temporal_masked_only", "spectral_masked_only", "spectral_all"]] = ["spectral_masked_only"],
        P: Union[List[float],float] = 2, #the p-norm for the losses
        loss_weights:Union[List[float], Literal["equal","balancing_0th_order"]] = "equal", #ToDO: allow for balancing_0th_order and potentially balancing_1st_order
        num_channels:int = 1, #number of channels in the input
        log_spectogram:bool = True,
        
        bottleneck_dim:int=256,
        bottleneck_activation:str="Identity",
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
        norm_method:Literal["patch", "channel", "global"] = "patch",
        **kwargs):

        # print(sequence_len)  
        self.spectogram_size = (n_fft//2 + 1, sequence_len//hop_length + 1)
        print(self.spectogram_size)
        self.reshape_size = max(self.spectogram_size) if reshape_size is None else reshape_size
        
        if reshape_size is None:
            self.interpolation_transform = "nearest"
            self.interpolation_inverse = "nearest"
        else:
            self.interpolation_transform = interpolation[0] if isinstance(interpolation, tuple) else interpolation
            self.interpolation_inverse = interpolation[1] if isinstance(interpolation, tuple) else interpolation
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
            num_channels= num_channels * (1 if not predict_phases else 2),
            qkv_bias=qkv_bias,
            decoder_num_attention_heads=decoder_num_attention_heads,
            decoder_hidden_size=decoder_hidden_size,
            decoder_num_hidden_layers=decoder_num_hidden_layers,
            decoder_intermediate_size=decoder_intermediate_size,
            mask_ratio=mask_ratio,
            norm_pix_loss=norm_pix_loss,
            **kwargs
        )
        self.bottleneck_dim = bottleneck_dim
        self.bottleneck_activation = bottleneck_activation

        self.losses = losses
        self.loss_weights = loss_weights
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.predict_phases = predict_phases
        self.sequence_len = sequence_len
        self.P = P
        self.log_spectogram = log_spectogram
        self.norm_method = norm_method  
        self.input_num_channels = num_channels


# @dataclass
# class AdditionalFoundationalModelOutput:
#     """Additional output from the foundational model

#     losses:dict[str, tuple[torch.FloatTensor, float]]: the name, loss and weight of the loss
#     """
#     losses:Optional[Dict[str, Tuple[float, float]]]

@dataclass
class ViTMAEForEMG_PretrainingOutput(FoundationalModelOutput):
    """
    Class for ViTMAEForPreTraining's outputs, with potential hidden states and attentions.

    Following from Huggingface's transformers:
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`):
            Pixel reconstruction loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    logits: torch.FloatTensor = None
    mask: torch.LongTensor = None
    ids_restore: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class ViTMAEForEMG_Pretraining(ViTMAEForPreTraining, ParentModel, ViTMAEPreTrainedModel):

    encoder_only_mode:bool = False
    def __init__(self, config: ViTMAEForEMGConfig):

        ViTMAEPreTrainedModel.__init__(self, config)
        self.config = config


        self.vit = ViTMAEModel(config)
        self.decoder = ViTMAEDecoder(config, num_patches=self.vit.embeddings.num_patches)
        #add a bottleneck layer
        self.bottleneck = nn.Linear(config.hidden_size, config.bottleneck_dim)
        self.bottleneck_activation = getattr(nn, config.bottleneck_activation)()

        #modify the embedding layer of the decoder to take in the bottleneck layer
        self.decoder.decoder_embed = nn.Linear(config.bottleneck_dim, config.decoder_hidden_size)

        # Initialize weights and apply final processing
        self.post_init()
        self.forward_transform = SpectogramTransform(
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            reshape_size=config.reshape_size,
            predict_phases=config.predict_phases,
            log = config.log_spectogram,
            interpolation=config.interpolation_transform
        )

        self.inverse_transform = InverseSpectogramTransform(
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            original_size=config.spectogram_size,
            predict_phases=config.predict_phases,
            log = config.log_spectogram,
            interpolation=config.interpolation_inverse
        )
        
        self.predict_phases = config.predict_phases

        self.losses = config.losses


        if isinstance(config.loss_weights, list):
            self.loss_weights = {loss: weight/sum(config.loss_weights) for loss, weight in zip(config.losses, config.loss_weights)}
            self.loss_weights_type = "constant"
        elif config.loss_weights == "equal":
            self.loss_weights = {loss: 1/len(config.losses) for loss in config.losses}
            self.loss_weights_type = "constant"
        else:
            #todo implement balancing_0th_order
            raise ValueError(f"Unknown loss weight configuration {config.loss_weights}")
    
        self.P = config.P if isinstance(config.P, list) else [config.P] * len(config.losses)
        assert len(self.P) == len(config.losses), "P must be the same length as the losses"

        self.sequence_len = config.sequence_len

    # @staticmethod
    def denorm_patches(self,predicted_patches: torch.FloatTensor, input_image_patches: torch.FloatTensor) -> torch.FloatTensor:

        #denorm the predicted patches
        if self.config.norm_method == "patch":
            mean = input_image_patches.mean(dim=-1, keepdim=True)
            var = input_image_patches.var(dim=-1, keepdim=True)
        elif self.config.norm_method == "channel":
            raise NotImplementedError("Channel normalization not implemented yet")
        elif self.config.norm_method == "global":
            mean = input_image_patches.mean(dim = (-2,-1), keepdim=True)
            var = input_image_patches.var(dim = (-2,-1), keepdim=True)
        return predicted_patches * (var + 1.0e-6) ** 0.5 + mean

    # @staticmethod
    def norm_input(self,input_image_patches: torch.FloatTensor) -> torch.FloatTensor:
        if self.config.norm_method == "patch":
            mean = input_image_patches.mean(dim=-1, keepdim=True)
            var = input_image_patches.var(dim=-1, keepdim=True)
        elif self.config.norm_method == "channel":
            raise NotImplementedError("Channel normalization not implemented yet")
        elif self.config.norm_method == "global":
            mean = input_image_patches.mean(dim = (-2,-1), keepdim=True)
            var = input_image_patches.var(dim = (-2,-1), keepdim=True)
        return (input_image_patches - mean) / (var + 1.0e-6) ** 0.5

    def temporal_loss_all(self, predicted_patches: torch.FloatTensor,
                            mask: torch.FloatTensor,
                            input_image: torch.FloatTensor,
                            input_sequence: torch.FloatTensor,
                            phases: Optional[torch.FloatTensor] = None,
                            interpolate_pos_encoding: Optional[bool] = False,
                            denorm:bool = True, #denorm is needed
                            p:Optional[int]=2
                             ) -> torch.FloatTensor:
        """temporal loss function

        Args:
            predicted_patches (torch.FloatTensor): shape of (batch_size, num_patches, patch_size**2 * num_channels)
            mask (torch.FloatTensor): shape of (batch_size, num_patches), Tensor indicating which patches are masked (1) and which are not (0).
            input_image (torch.FloatTensor): shape of (batch_size, num_channels, height, width)
            input_sequence (torch.FloatTensor): shape of (.., sequence_len)
            phases (Optional[torch.FloatTensor], optional): the phase spectrogram of shape (..., 1, n_fft//2 + 1, sequence_len//hop_length + 1). 
                    Defaults to None, necessary if predict_phases is False.
            interpolate_pos_encoding (bool, optional): whether to interpolate the positional encoding. Defaults to False.
            denorm (bool, optional): whether to denorm the predicted patches. Defaults to True.
            p (Optional[int], optional): the p-norm of the loss. Defaults to 2.

        Returns:
            torch.FloatTensor: MSE loss between the predicted waveform and the target waveform
        """

        if self.config.norm_pix_loss and denorm:
            predicted_patches = self.denorm_patches(predicted_patches, self.patchify(input_image, interpolate_pos_encoding=interpolate_pos_encoding))

        #convert the patches to the original image size
        predicted_spectogram = self.unpatchify(predicted_patches)
        #convert the predicted spectogram to the waveform
        predicted_waveform = self.inverse_transform(predicted_spectogram, phases)
        #calculate the loss
        loss = temporal_loss(predicted_waveform, input_sequence, p=p)
        return loss
    
    def temporal_loss_masked(self, predicted_patches: torch.FloatTensor,
                            mask: torch.FloatTensor,
                            input_image: torch.FloatTensor,
                            input_sequence: torch.FloatTensor,
                            phases: Optional[torch.FloatTensor] = None,
                            interpolate_pos_encoding: Optional[bool] = False,
                            p:Optional[int]=2) -> torch.FloatTensor:
        """temporal loss function which only the masked patches are used to reconstruct the waveform,
        ie the temporal analog the normal ViTMAE loss

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
            #we want to denorm the predicted patches
            predicted_patches = self.denorm_patches(predicted_patches, input_patches)
        
        patches = predicted_patches * mask.unsqueeze(-1) + input_patches * (1 - mask).unsqueeze(-1)
        
        return self.temporal_loss_all(patches, mask, input_image, input_sequence, phases, interpolate_pos_encoding, denorm=False,
                                        p=p)
    
    def spectral_loss_all(self, predicted_patches: torch.FloatTensor, input_image:torch.FloatTensor,
                          interpolate_pos_encoding: Optional[bool] = False,
                          p:Optional[int]=2
                          ) -> torch.FloatTensor:
        """spectral loss for all patches masked or not

        Args:
            predicted_patches (torch.FloatTensor): shape of (batch_size, num_patches, patch_size**2 * num_channels)
            input_image (torch.FloatTensor): shape of (batch_size, num_channels, height, width)
            interpolate_pos_encoding (bool, optional): whether to interpolate the positional encoding. Defaults to False.
            p (Optional[int], optional): the p-norm of the loss. Defaults to 2.
        
        Returns:
            torch.FloatTensor: MSE loss between the predicted patches and the target patches
        """

        input_patches = self.patchify(input_image, interpolate_pos_encoding=interpolate_pos_encoding)

        if self.config.norm_pix_loss:
            input_patches = self.norm_input(input_patches)

        return spectral_loss(predicted_patches, input_patches, p=self.config.P)
    
    def spectral_loss_masked(self, predicted_patches: torch.FloatTensor,
                            mask: torch.FloatTensor,
                            input_image: torch.FloatTensor,    
                            interpolate_pos_encoding: Optional[bool] = False,
                            p:Optional[int]=2
                                ) -> torch.FloatTensor:
        
        """spectral loss for masked patches

        Args:
            predicted_patches (torch.FloatTensor): shape of (batch_size, num_patches, patch_size**2 * num_channels)
            mask (torch.FloatTensor): shape of (batch_size, num_patches), Tensor indicating which patches are masked (1) and which are not (0).
            input_image (torch.FloatTensor): shape of (batch_size, num_channels, height, width)
            interpolate_pos_encoding (bool, optional): whether to interpolate the positional encoding. Defaults to False.
            p (Optional[int], optional): the p-norm of the loss. Defaults to 2.
        
        Returns:
            torch.FloatTensor: MSE loss between the predicted patches and the target patches
        """

        input_patches = self.patchify(input_image, interpolate_pos_encoding=interpolate_pos_encoding)

        if self.config.norm_pix_loss:
            input_patches = self.norm_input(input_patches)
        
        return spectral_loss(predicted_patches, input_patches, mask, p=p)
        

    
    def forward_loss(self, predicted_patches:torch.FloatTensor,
                     mask: torch.FloatTensor,   
                     input_image: torch.FloatTensor,
                     input_sequence: torch.FloatTensor,
                     phases: Optional[torch.FloatTensor] = None,
                     interpolate_pos_encoding: Optional[bool] = False) -> torch.FloatTensor:
        
        """calculate the losses"""

        loss = 0
        loss_dict = {}
        for i,loss_name in enumerate(self.losses):
            weight = self.loss_weights[loss_name]
            p = self.P[i]
            if loss_name == "temporal_all":
                # print("predicted_patches.shape", predicted_patches.shape)
                l = self.temporal_loss_all(predicted_patches, mask, input_image, input_sequence, phases, interpolate_pos_encoding, p = p) 
            elif loss_name == "temporal_masked_only":
                # print("predicted_patches.shape", predicted_patches.shape)
                l = self.temporal_loss_masked(predicted_patches, mask, input_image, input_sequence, phases, interpolate_pos_encoding, p = p) 
            elif loss_name == "spectral_masked_only":
                l = self.spectral_loss_masked(predicted_patches, mask, input_image, interpolate_pos_encoding, p = p)
            elif loss_name == "spectral_all":
                l = self.spectral_loss_all(predicted_patches, input_image, interpolate_pos_encoding, p = p)
            else:
                raise ValueError(f"Unknown loss {loss_name}")
            if weight > 0:
                loss += weight * l #allows us to use some losses for logging only
            loss_dict[loss_name] = (l.item(), weight)
        
        if self.loss_weights_type != "constant":
            raise NotImplementedError("Balancing methods not implemented yet")
        
        return loss, loss_dict
        

    def forward(
        self,
        input_waveforms: torch.FloatTensor,
        noise: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Union[Tuple, ViTMAEForPreTrainingOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.predict_phases:
            input_specs = self.forward_transform(input_waveforms)
            phases = None
        else:
            
            input_specs,phases = self.forward_transform(input_waveforms)
        #pixel values are of shape (batch_size, num_channels, 2|1, height, width)

        outputs = self.vit(
            input_specs,
            noise=noise,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        latent = outputs.last_hidden_state
        #push the latent representation through the bottleneck layer
        latent = self.bottleneck(latent)
        latent = self.bottleneck_activation(latent)
        if self.encoder_only_mode:
            return latent.reshape(latent.shape[0], -1) #flatten the latent representation
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        decoder_outputs = self.decoder(latent, ids_restore, interpolate_pos_encoding=interpolate_pos_encoding)
        logits = decoder_outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)

        loss, loss_dict = self.forward_loss(logits, mask, input_specs, input_waveforms, phases, interpolate_pos_encoding)
        # raise ValueError("stop here")
        if not return_dict:
            output = ((logits, mask, ids_restore) + outputs[2:] + (loss_dict)) if loss is not None else (logits, mask, ids_restore) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ViTMAEForEMG_PretrainingOutput(
            loss=loss,
            losses=loss_dict,
            logits=logits,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            phases = phases,
            input_specs = input_specs,
            input_waveforms = input_waveforms
        )
    
    def reconstruct_spectograms(self, model_output: ViTMAEForEMG_PretrainingOutput) -> List[Tuple[List[torch.FloatTensor], List[str]]]:
        """Reconstruct the spectograms from the model output
        Args:
            model_output (ViTMAEForEMG_PretrainingOutput): the model output
        Returns:
            List[Tuple[List[torch.FloatTensor], List[str]]]: the reconstructed spectograms and their names
            return a seperate Tuple for entry in the batch, as they will be plotted seperately
        """
        
        output = []

        #get the input spectograms
        input_specs = model_output.input_specs.detach().cpu() #shape of (batch_size, n_channels * (2|1), height, width)
        #get the model output
        reconstructed_patches = model_output.logits #shape of (batch_size, num_patches, patch_size*patch_size*num_channels)
        mask = model_output.mask #shape of (batch_size, num_patches)

        #unpatchify the patches
        reconstructed_specs = self.unpatchify(reconstructed_patches).detach().cpu()
        #check that it is the same size as the input specs
        assert reconstructed_specs.shape == input_specs.shape, "The reconstructed spectogram is not the same size as the input spectogram"
        #denorm the patches
        if self.config.norm_pix_loss:
            reconstructed_specs = self.denorm_patches(reconstructed_specs, input_specs)
        
        #depatchify the mask
        mask = self.unpatchify(mask.unsqueeze(-1).repeat(1,1,reconstructed_patches.shape[-1])).detach().cpu() #shape of (batch_size, n_channels * (2|1), height, width)

        # #flatten the 2nd and 3rd dimensions for all the tensors
        # print(input_specs.shape)
        # # input_specs = input_specs.reshape(input_specs.shape[0], -1, input_specs.shape[-3:]).detach().cpu()
        # # reconstructed_specs = reconstructed_specs.reshape(reconstructed_specs.shape[0], -1, reconstructed_specs.shape[-3:]).detach().cpu()
        # # mask = mask.reshape(mask.shape[0], -1, mask.shape[-3:]).detach().cpu() #not sure if we need to detach \shrug lol

        channel_names = ["mag"] if not self.predict_phases else ["mag", "phase"]

        #for each batch
        for i in range(input_specs.shape[0]):
            output_i = []
            names_i = []
            for j in range(input_specs.shape[1]):
                #for each channel
                #add 3 things, the input spectogram, the masked input spectogram, and the masked input spectogram
                masked_input = input_specs[i,j].clone()
                #set the masked (1.0) value to be nan
                masked_input[mask[i,j] == 1.0] = torch.nan
                output_i += [input_specs[i,j], masked_input, reconstructed_specs[i,j]]
                prefix = f"channel_{j//len(channel_names)}_{channel_names[j%len(channel_names)]}"
                names_i += [f"{prefix}_input", f"{prefix}_masked", f"{prefix}_reconstructed"]
            output.append((output_i, names_i))
        
        return output
    
    def reconstruct_waveforms(self,
                                model_output: ViTMAEForEMG_PretrainingOutput) -> List[Tuple[torch.FloatTensor, str]]:
        """Reconstruct the waveforms from the model output
        Args:
            model_output (ViTMAEForEMG_PretrainingOutput): the model output
        Returns:
            List[Tuple[torch.FloatTensor, str]]: the reconstructed waveforms and their names
            once again, return a seperate Tuple for entry in the batch, as they will be plotted seperately
        """
        output = []

        #get the input waveforms
        input_waveforms = model_output.input_waveforms #shape of (batch_size, n_channels, sequence_len)
        #unpatchify the model output, and possibly denorm it
        reconstructed_patches = model_output.logits
        #unpatchify the patches
        reconstructed_specs = self.unpatchify(reconstructed_patches)
        
        #denorm if needed
        if self.config.norm_pix_loss:
            reconstructed_specs = self.denorm_patches(reconstructed_specs, model_output.input_specs)
        
        #convert the spectogram to the waveform
        reconstructed_waveforms = self.inverse_transform(reconstructed_specs, phases = model_output.phases) #shape of (batch_size, n_channels, sequence_len)

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
    
    def encoder_only(self, freeze_encoder:bool):

        #delete the decoder
        del self.decoder

        #clean memory
        torch.cuda.empty_cache()

        self.encoder_only_mode = True

        if freeze_encoder:
            for param in self.vit.parameters():
                param.requires_grad = False
            for param in self.bottleneck.parameters():
                param.requires_grad = False
        else:
            for param in self.vit.parameters():
                param.requires_grad = True
            for param in self.bottleneck.parameters():
                param.requires_grad = True

    def get_encoder_output_size(self):
        return self.config.bottleneck_dim * (self.config.reshape_size // self.config.patch_size) ** 2
            
        


class ViTMAE_Pretraining_Lightning(Pretraining_Lightning):
    def __init__(self, LightningConfig: LightningConfig):
                 
        super().__init__(LightningConfig)
        self.model = ViTMAEForEMG_Pretraining(LightningConfig.model_config)

    
if __name__ == "__main__":

    example_emg = torch.randn(1, 10000)

    model = ViTMAEForEMG_Pretraining(ViTMAEForEMGConfig(sequence_len = example_emg.shape[-1],
                                                        losses = ["temporal_masked_only","spectral_masked_only"],
                                                        loss_weights = "equal"))

    output = model(example_emg)

    print(output.loss)
    print(output.loss_dict) 

