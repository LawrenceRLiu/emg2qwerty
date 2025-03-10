from transformers.models.vit_mae.modeling_vit_mae import ViTMAEDecoder
from foundational_model.models.ViTMAE import ViTMAEForEMG_Pretraining
from transformers import ViTMAEConfig, ViTMAEModel, ViTMAEPreTrainedModel

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

from ..utils.loss import temporal_loss, spectral_loss
from ..utils.transforms import SpectogramTransform, InverseSpectogramTransform
from ..utils.parent_lightning_pretrainer import LightningConfig, FoundationalModelOutput, ParentModel, Pretraining_Lightning


class ViTVAEForEMGConfig(ViTMAEConfig):
    def __init__(
        self,
        #spectogram configs
        sequence_len: int = 1000,
        n_fft: int = 128,
        hop_length: int = 64,
        reshape_size: Optional[int] = 128,
        interpolation:Union[Tuple[Literal["linear", "nearest", "cubic"]],Literal["linear", "nearest", "cubic"]] = "linear", #supports different resampling methods for transform and inverse transform
        predict_phases: bool = False, #predict the phases or just pass them along
        losses: List[Literal["temporal_all", "temporal_masked_only", "spectral_masked_only", "spectral_all"]] = ["spectral_masked_only"],
        P: Union[List[float], float] = 2, #the p-norm for the losses
        loss_weights: Union[List[float], Literal["equal", "balancing_0th_order"]] = "equal", #ToDO: allow for balancing_0th_order and potentially balancing_1st_order
        num_channels: int = 1, #number of channels in the input
        log_spectogram: bool = True,
        
        latent_dim: int = 256,
        latent_activation: str = "Identity",
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        patch_size: int = 16,
        qkv_bias: bool = True,
        decoder_num_attention_heads: int = 16,
        decoder_hidden_size: int = 512,
        decoder_num_hidden_layers: int = 8,
        decoder_intermediate_size: int = 2048,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = False,
        norm_method: Literal["patch", "channel", "global"] = "patch",
        **kwargs
    ):
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
            num_channels=num_channels * (1 if not predict_phases else 2),
            qkv_bias=qkv_bias,
            decoder_num_attention_heads=decoder_num_attention_heads,
            decoder_hidden_size=decoder_hidden_size,
            decoder_num_hidden_layers=decoder_num_hidden_layers,
            decoder_intermediate_size=decoder_intermediate_size,
            mask_ratio=mask_ratio,
            norm_pix_loss=norm_pix_loss,
            **kwargs,
        )

        self.bottleneck_dim = latent_dim
        self.bottleneck_activation = latent_activation

        self.losses = losses
        self.loss_weights = loss_weights
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.predict_phases = predict_phases
        self.sequence_len = sequence_len
        self.P = P
        self.log_spectogram = log_spectogram
        self.norm_method = norm_method


@dataclass
class ViTMAEForEMG_PretrainingOutput(FoundationalModelOutput):
    """
    Pretraining output for our ViTVAE EMG model extended to include KL divergence loss
    """
    loss: Optional[torch.FloatTensor] = None
    losses: Optional[Dict[str, Tuple[float, float]]] = None
    logits: Optional[torch.FloatTensor] = None
    mask: Optional[torch.LongTensor] = None
    ids_restore: Optional[torch.LongTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    phases: Optional[torch.FloatTensor] = None
    input_specs: Optional[torch.FloatTensor] = None
    input_waveforms: Optional[torch.FloatTensor] = None
    mu: Optional[torch.FloatTensor] = None
    logvar: Optional[torch.FloatTensor] = None


class ViTVAEForEMG(ViTMAEPreTrainedModel):
    """
    Inherits from ViTMAE encoder and decoder and compute latent distribution
    """
    config_class = ViTVAEForEMGConfig

    def __init__(self, config: ViTVAEForEMGConfig):
        super().__init__(config)
        #reuse the ViTMAE encoder
        self.encoder = ViTMAEModel(config)
        self.num_tokens = (config.image_size//config.patch_size) ** 2
        self.pool = nn.AdaptiveAvgPool1d(1)
        #project global representation to latent distribution parameters
        self.fc_mu = nn.Linear(config.hidden_size, config.latent_dim)
        self.fc_logvar = nn.Linear(config.hidden_size, config.latent_dim)
        self.fc_z = nn.Linear(config.latent_dim, config.hidden_size * self.num_tokens)
        #reuse ViTMAE decoder
        self.decoder = ViTMAEDecoder(config)
        self.init_weights()

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> ViTMAEForEMG_PretrainingOutput:
        """calculate the losses"""
        encoder_outputs = self.encoder(pixel_values, **kwargs)
        hidden_states = encoder_outputs.last_hidden_state
        global_repr = hidden_states.mean(dim=1)
        mu = self.fc_mu(global_repr)
        logvar = self.fc_logvar(global_repr)
        z = self.reparameterize(mu, logvar)
        z_proj = self.fc_z(z)
        z_proj = z_proj.view(-1, self.num_tokens, self.encoder.config.hidden_size)
        decoder_outputs = self.decoder(x=z_proj, ids_restore=None, output_hidden_states=kwargs.get("output_hidden_states", False))
        reconstruction_logits = decoder_outputs.logits
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_loss = kl_loss.mean()

        return ViTMAEForEMG_PretrainingOutput(
            logits=reconstruction_logits,
            loss=None,
            losses=None,
            mask=encoder_outputs.mask if hasattr(encoder_outputs, "mask") else None,
            ids_restore=encoder_outputs.ids_restore if hasattr(encoder_outputs, "ids_restore") else None,
            hidden_states=decoder_outputs.hidden_states if hasattr(decoder_outputs, "hidden_states") else None,
            attentions=decoder_outputs.attentions if hasattr(decoder_outputs, "attentions") else None,
            mu=mu,
            logvar=logvar,
        )


class ViTVAEForEMG_Pretraining(ParentModel, ViTMAEPreTrainedModel):
    def __init__(self, config: ViTVAEForEMGConfig):
        ViTMAEPreTrainedModel.__init__(self, config)
        self.config = config
        self.vtv = ViTVAEForEMG(config)
        self.forward_transform = SpectogramTransform(
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            reshape_size=config.reshape_size,
            predict_phases=config.predict_phases,
            log=config.log_spectogram,
            interpolation=config.interpolation_transform
        )
        self.inverse_transform = InverseSpectogramTransform(
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            original_size=config.spectogram_size,
            predict_phases=config.predict_phases,
            log=config.log_spectogram,
            interpolation=config.interpolation_inverse
        )
        self.predict_phases = config.predict_phases
        self.losses = config.losses
        if isinstance(config.loss_weights, list):
            self.loss_weights = {loss: weight / sum(config.loss_weights) for loss, weight in zip(config.losses, config.loss_weights)}
            self.loss_weights_type = "constant"
        elif config.loss_weights == "equal":
            self.loss_weights = {loss: 1 / len(config.losses) for loss in config.losses}
            self.loss_weights_type = "constant"
        else:
            #todo implement balancing_0th_order
            raise ValueError(f"Unknown loss weight configuration {config.loss_weights}")
        self.P = config.P if isinstance(config.P, list) else [config.P] * len(config.losses)
        assert len(self.P) == len(config.losses), "P must be the same length as the losses"
        self.sequence_len = config.sequence_len

        self.post_init()

    def patchify(self, imgs: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        patch_size = self.config.patch_size
        B, C, H, W = imgs.shape
        assert H == self.config.image_size and W == self.config.image_size, "Image size mismatch with config"
        patches = imgs.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, patch_size * patch_size * C)
        return patches

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        patch_size = self.config.patch_size
        B, N, patch_dim = patches.shape
        H = W = self.config.image_size
        h = H//patch_size
        w = W//patch_size
        patches = patches.reshape(B, h, w, -1)
        imgs = patches.permute(0, 3, 1, 2)
        return imgs

    def norm_input(self, input_image_patches: torch.FloatTensor) -> torch.FloatTensor:
        if self.config.norm_method == "patch":
            mean = input_image_patches.mean(dim=-1, keepdim=True)
            var = input_image_patches.var(dim=-1, keepdim=True)
        elif self.config.norm_method == "channel":
            raise NotImplementedError("Channel normalization not implemented yet")
        elif self.config.norm_method == "global":
            mean = input_image_patches.mean(dim=(-2, -1), keepdim=True)
            var = input_image_patches.var(dim=(-2, -1), keepdim=True)
        return (input_image_patches - mean) / (var + 1.0e-6) ** 0.5

    def denorm_patches(self, predicted_patches: torch.FloatTensor, input_image_patches: torch.FloatTensor) -> torch.FloatTensor:
        if self.config.norm_method == "patch":
            mean = input_image_patches.mean(dim=-1, keepdim=True)
            var = input_image_patches.var(dim=-1, keepdim=True)
        elif self.config.norm_method == "channel":
            raise NotImplementedError("Channel normalization not implemented yet")
        elif self.config.norm_method == "global":
            mean = input_image_patches.mean(dim=(-2, -1), keepdim=True)
            var = input_image_patches.var(dim=(-2, -1), keepdim=True)
        return predicted_patches * (var + 1.0e-6) ** 0.5 + mean

    def temporal_loss_all(self, predicted_patches: torch.FloatTensor,
                          mask: Optional[torch.FloatTensor],
                          input_image: torch.FloatTensor,
                          input_sequence: torch.FloatTensor,
                          phases: Optional[torch.FloatTensor] = None,
                          interpolate_pos_encoding: Optional[bool] = False,
                          denorm: bool = True,
                          p: Optional[int] = 2) -> torch.FloatTensor:
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
            predicted_patches = self.denorm_patches(predicted_patches,
                                                     self.patchify(input_image, interpolate_pos_encoding))
        #convert the patches to the original image size
        predicted_spectrogram = self.unpatchify(predicted_patches)
        #convert the predicted spectogram to the waveform
        predicted_waveform = self.inverse_transform(predicted_spectrogram, phases)
        #calculate the loss
        loss = temporal_loss(predicted_waveform, input_sequence, p=p)
        return loss

    def temporal_loss_masked(self, predicted_patches: torch.FloatTensor,
                             mask: torch.FloatTensor,
                             input_image: torch.FloatTensor,
                             input_sequence: torch.FloatTensor,
                             phases: Optional[torch.FloatTensor] = None,
                             interpolate_pos_encoding: Optional[bool] = False,
                             p: Optional[int] = 2) -> torch.FloatTensor:
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
        input_patches = self.patchify(input_image, interpolate_pos_encoding)
        if self.config.norm_pix_loss:
            #we want to denorm the predicted patches
            predicted_patches = self.denorm_patches(predicted_patches, input_patches)
        patches = predicted_patches * mask.unsqueeze(-1) + input_patches * (1 - mask).unsqueeze(-1)
        return self.temporal_loss_all(patches, mask, input_image, input_sequence, phases, interpolate_pos_encoding, denorm=False, p=p)

    def spectral_loss_all(self, predicted_patches: torch.FloatTensor, input_image: torch.FloatTensor,
                          interpolate_pos_encoding: Optional[bool] = False,
                          p: Optional[int] = 2) -> torch.FloatTensor:
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
                             p: Optional[int] = 2) -> torch.FloatTensor:
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

    def forward_loss(self, predicted_patches: torch.FloatTensor,
                     mask: torch.FloatTensor,
                     input_image: torch.FloatTensor,
                     input_sequence: torch.FloatTensor,
                     phases: Optional[torch.FloatTensor] = None,
                     interpolate_pos_encoding: Optional[bool] = False) -> Tuple[torch.FloatTensor, Dict[str, Tuple[float, float]]]:
        """calculate the losses"""
        loss = 0
        loss_dict = {}
        for i, loss_name in enumerate(self.losses):
            weight = self.loss_weights[loss_name]
            p_val = self.P[i]
            if loss_name == "temporal_all":
                # print("predicted_patches.shape", predicted_patches.shape)
                l = self.temporal_loss_all(predicted_patches, mask, input_image, input_sequence, phases, interpolate_pos_encoding, p=p_val)
            elif loss_name == "temporal_masked_only":
                # print("predicted_patches.shape", predicted_patches.shape)
                l = self.temporal_loss_masked(predicted_patches, mask, input_image, input_sequence, phases, interpolate_pos_encoding, p=p_val)
            elif loss_name == "spectral_masked_only":
                l = self.spectral_loss_masked(predicted_patches, mask, input_image, interpolate_pos_encoding, p=p_val)
            elif loss_name == "spectral_all":
                l = self.spectral_loss_all(predicted_patches, input_image, interpolate_pos_encoding, p=p_val)
            else:
                raise ValueError(f"Unknown loss {loss_name}")
            if weight > 0:
                loss += weight * l  #allows us to use some losses for logging only
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
    ) -> Union[Tuple, ViTMAEForEMG_PretrainingOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.predict_phases:
            input_specs = self.forward_transform(input_waveforms)
            phases = None
        else:
            input_specs, phases = self.forward_transform(input_waveforms)
        #pixel values are of shape (batch_size, num_channels, 2|1, height, width)
        outputs = self.vtv(input_specs,
                           noise=noise,
                           head_mask=head_mask,
                           output_attentions=output_attentions,
                           output_hidden_states=output_hidden_states,
                           return_dict=return_dict,
                           interpolate_pos_encoding=interpolate_pos_encoding)
        predicted_patches = outputs.logits  # (B, num_tokens, patch_dim)
        mask = outputs.mask
        ids_restore = outputs.ids_restore

        #compute temporal, spetral and kl divergence loss
        recon_loss, loss_dict = self.forward_loss(predicted_patches, mask, input_specs, input_waveforms, phases, interpolate_pos_encoding)
        total_loss = recon_loss + outputs.loss if outputs.loss is not None else recon_loss
        total_loss = total_loss + outputs.kl_loss
        loss_dict["kl"] = (outputs.kl_loss.item(), 1.0)

        if not return_dict:
            output = ((predicted_patches, mask, ids_restore) + outputs[2:] + (loss_dict,)) if total_loss is not None else (predicted_patches, mask, ids_restore) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return ViTMAEForEMG_PretrainingOutput(
            loss=total_loss,
            losses=loss_dict,
            logits=predicted_patches,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            phases=phases,
            input_specs=input_specs,
            input_waveforms=input_waveforms,
            mu=outputs.mu,
            logvar=outputs.logvar,
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
        reconstructed_patches = model_output.logits.detach().cpu() #shape of (batch_size, num_patches, patch_size*patch_size*num_channels)
        mask = model_output.mask.detach().cpu() #shape of (batch_size, num_patches)

        #unpatchify the patches
        reconstructed_specs = self.unpatchify(reconstructed_patches)
        #check that it is the same size as the input specs
        assert reconstructed_specs.shape == input_specs.shape, "Mismatch in reconstructed spectrogram shape"
        #denorm the patches
        if self.config.norm_pix_loss:
            reconstructed_specs = self.denorm_patches(reconstructed_specs, input_specs)
        #depatchify the mask
        mask = self.unpatchify(mask.unsqueeze(-1).repeat(1, 1, reconstructed_patches.shape[-1])).detach().cpu()
        channel_names = ["mag"] if not self.predict_phases else ["mag", "phase"]

        #for each batch
        for i in range(input_specs.shape[0]):
            output_i = []
            names_i = []
            for j in range(input_specs.shape[1]):
                #for each channel
                #add 3 things, the input spectogram, the masked input spectogram, and the masked input spectogram
                masked_input = input_specs[i, j].clone()
                #set the masked (1.0) value to be nan
                masked_input[mask[i, j] == 1.0] = torch.nan
                output_i += [input_specs[i, j], masked_input, reconstructed_specs[i, j]]
                prefix = f"channel_{j//len(channel_names)}_{channel_names[j % len(channel_names)]}"
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
        reconstructed_waveforms = self.inverse_transform(reconstructed_specs, phases=model_output.phases)
        
        #similar to the spectograms
        for i in range(input_waveforms.shape[0]):
            output_i = []
            names_i = []
            for j in range(input_waveforms.shape[1]):
                output_i.append(input_waveforms[i, j].cpu())
                output_i.append(reconstructed_waveforms[i, j].detach().cpu())
                names_i.append(f"channel_{j}_input")
                names_i.append(f"channel_{j}_reconstructed")
            output.append((output_i, names_i))
        
        return output


class ViTVAE_Pretraining_Lightning(Pretraining_Lightning):
    def __init__(self, LightningConfig: LightningConfig):
        super().__init__(LightningConfig)
        self.model = ViTMAEForEMG_Pretraining(LightningConfig.model_config)


if __name__ == "__main__":
    example_emg = torch.randn(1, 10000)

    model = ViTVAEForEMG_Pretraining(ViTVAEForEMGConfig(sequence_len=example_emg.shape[-1],
                                                        losses=["temporal_masked_only", "spectral_masked_only"],
                                                        loss_weights="equal"))

    output = model(example_emg)

    print(output.loss)
    print(output.losses)
