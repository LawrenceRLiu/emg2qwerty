from transformers import ViTMAEConfig, ViTMAEModel, ViTMAEForPreTraining

from transformers.models.vit_mae.modeling_vit_mae import ViTMAEForPreTrainingOutput

import torch
import torch.nn as nn
from dataclasses import dataclass
import lightning as L
from typing import Any, Dict, List, Tuple, Union, Optional, Callable, Literal

from ..utils.loss import temporal_loss, spectral_loss
from ..utils.transforms import SpectogramTransform, InverseSpectogramTransform
from ..utils.custom_logger import CustomPretrainLogger

from .output import AdditionalFoundationalModelOutput
#call the model ViTMAEForEMG


class ViTMAEForEMGConfig(ViTMAEConfig):
    def __init__(self, 
        #spectogram configs
        sequence_len:int=1000,
        n_fft:int = 128,
        hop_length:int = 64,
        predict_phases:bool = False, #predict the phases or just pass them along
        losses:List[Literal["temporal_all", "temporal_masked_only", "spectral_masked_only", "spectral_all"]] = ["spectral_masked_only"],
        P: Union[List[float],float] = 2, #the p-norm for the losses
        loss_weights:Union[List[float], Literal["equal","balancing_0th_order"]] = "equal", #ToDO: allow for balancing_0th_order and potentially balancing_1st_order
        num_channels:int = 1, #number of channels in the input
        log_spectogram:bool = True,

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

        # print(sequence_len)  
        self.spectogram_size = (n_fft//2 + 1, sequence_len//hop_length + 1)
        # print(self.spectogram_size)
        self.reshape_size = max(self.spectogram_size)
        #increase until its a multiple of patch_size
        while self.reshape_size % patch_size != 0:
            self.reshape_size += 1
        # print(self.reshape_size)
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

        self.losses = losses
        self.loss_weights = loss_weights
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.predict_phases = predict_phases
        self.sequence_len = sequence_len
        self.P = P
        self.log_spectogram = log_spectogram

    


@dataclass
class ViTMAEForEMG_PretrainingOutput(ViTMAEForPreTrainingOutput, AdditionalFoundationalModelOutput):
    phases: Optional[torch.FloatTensor] = None
    input_specs: Optional[torch.FloatTensor] = None
    input_waveforms: Optional[torch.FloatTensor] = None

class ViTMAEForEMG_Pretraining(ViTMAEForPreTraining):


    def __init__(self, config: ViTMAEForEMGConfig):
        super().__init__(config)

        self.forward_transform = SpectogramTransform(
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            reshape_size=config.reshape_size,
            predict_phases=config.predict_phases,
            log = config.log_spectogram
        )

        self.inverse_transform = InverseSpectogramTransform(
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            original_size=config.spectogram_size,
            predict_phases=config.predict_phases,
            log = config.log_spectogram
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

    @staticmethod
    def denorm_patches(predicted_patches: torch.FloatTensor, input_image_patches: torch.FloatTensor) -> torch.FloatTensor:

        #denorm the predicted patches
        mean = input_image_patches.mean(dim=-1, keepdim=True)
        var = input_image_patches.var(dim=-1, keepdim=True)
        return predicted_patches * (var + 1.0e-6) ** 0.5 + mean

    @staticmethod
    def norm_input(input_image_patches: torch.FloatTensor) -> torch.FloatTensor:
        mean = input_image_patches.mean(dim=-1, keepdim=True)
        var = input_image_patches.var(dim=-1, keepdim=True)
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
            predicted_patches = self.denorm_patches(predicted_patches, input_image)

        #convert the patches to the original image size
        predicted_spectogram = self.unpatchify(predicted_patches)
        #convert the predicted spectogram to the waveform
        predicted_waveform = self.inverse_transform(predicted_spectogram, phases)
        #calculate the loss
        loss = temporal_loss(predicted_waveform, input_sequence, mask, p=p)
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
        
    def patchify(self, input_image: torch.FloatTensor, interpolate_pos_encoding: Optional[bool] = False) -> torch.FloatTensor:
        """Patchify to handle the shape of the input image which is of shape (..., num_channels, 2|1, height, width)
        """
        #reshape the input by combining the -4 and -3 dimensions
        input_image = input_image.view(input_image.shape[:-4] + (-1,) + input_image.shape[-2:])
        
        #call the patchify function
        return super().patchify(input_image, interpolate_pos_encoding=interpolate_pos_encoding)

    def unpatchify(self, patches: torch.FloatTensor) -> torch.FloatTensor:
        """Unpatchify to handle the shape of the input image which is of shape (..., num_channels, 2|1, height, width)
        """
        #call the unpatchify function
        patches = super().unpatchify(patches)
        #reshape the patches to the original shape
        return patches.view(patches.shape[:-2] + (self.config.num_channels, -1) + patches.shape[-2:])

    
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
            p = self.P 
            if loss_name == "temporal_all":
                l = self.temporal_loss_all(predicted_patches, mask, input_image, input_sequence, phases, interpolate_pos_encoding, p = p) 
            elif loss_name == "temporal_masked_only":
                l = self.temporal_loss_masked(predicted_patches, mask, input_image, input_sequence, phases, interpolate_pos_encoding, p = p) 
            elif loss_name == "spectral_masked_only":
                l = self.spectral_loss_masked(predicted_patches, mask, input_image, interpolate_pos_encoding, p = p)
            elif loss_name == "spectral_all":
                l = self.spectral_loss_all(predicted_patches, input_image, interpolate_pos_encoding, p = p)
            else:
                raise ValueError(f"Unknown loss {loss_name}")
            
            loss += weight * l
            loss_dict[loss_name] = (l, weight)
        
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
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        decoder_outputs = self.decoder(latent, ids_restore, interpolate_pos_encoding=interpolate_pos_encoding)
        logits = decoder_outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)

        loss, loss_dict = self.forward_loss(logits, mask, input_specs, input_waveforms, phases, interpolate_pos_encoding)

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
            phases = phases,
            input_specs = input_specs,
            input_waveforms = input_waveforms
        )


class LightningConfig(ViTMAEForEMGConfig):
    def __init__(self, optimizer_name: str = "Adam",
                    lr: float = 1e-4,
                    scheduler_name: str = "LambdaLR",
                    scheduler_kwargs: Dict[str, Any] = {},
                    sample_log_interval: Union[int, Tuple[int,int]] = 100, 
                    #tuple allows us to pick different logging intervals for train and val
                    log_one_sample: bool = False,
                    #log one sample of the input and output spectograms or all the samples in a batch,
                    **kwargs):
            super().__init__(**kwargs)
            self.optimizer_name = optimizer_name
            self.lr = lr
            self.scheduler_name = scheduler_name
            self.scheduler_kwargs = scheduler_kwargs
            self.sample_log_interval_train = sample_log_interval[0] if isinstance(sample_log_interval, tuple) else sample_log_interval
            self.sample_log_interval_val = sample_log_interval[1] if isinstance(sample_log_interval, tuple) else sample_log_interval
                 

class ViTMAE_Pretraining_Lightning(L.LightningModule):
    def __init__(self, LightningConfig: LightningConfig):
                 
        super().__init__()
        self.model = ViTMAEForEMG_Pretraining(LightningConfig)
        self.config = LightningConfig
        self.logger: CustomPretrainLogger

    def forward(self, x: torch.FloatTensor) -> Any:
        return self.model(x)
    
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
        input_specs = model_output.input_specs #shape of (batch_size, n_channels, 2|1, height, width)
        #get the model output
        reconstructed_patches = model_output.logits #shape of (batch_size, num_patches, patch_size*patch_size*num_channels)
        mask = model_output.mask #shape of (batch_size, num_patches)

        #unpatchify the patches
        reconstructed_specs = self.model.unpatchify(reconstructed_patches)
        #check that it is the same size as the input specs
        assert reconstructed_specs.shape == input_specs.shape, "The reconstructed spectogram is not the same size as the input spectogram"
        #denorm the patches
        if self.model.config.norm_pix_loss:
            reconstructed_specs = self.model.denorm_patches(reconstructed_specs, input_specs)
        
        #depatchify the mask
        mask = self.model.unpatchify(mask.unsqueeze(-1).expand(-1, -1, input_specs.shape[-2] * input_specs.shape[-1])) #shape of (batch_size, n_channels, 2|1, height, width)

        #flatten the 2nd and 3rd dimensions for all the tensors
        input_specs = input_specs.reshape(input_specs.shape[0], -1, input_specs.shape[-3:]).detach().cpu()
        reconstructed_specs = reconstructed_specs.reshape(reconstructed_specs.shape[0], -1, reconstructed_specs.shape[-3:]).detach().cpu()
        mask = mask.reshape(mask.shape[0], -1, mask.shape[-3:]).detach().cpu() #not sure if we need to detach \shrug lol

        channel_names = ["mag"] if not self.model.predict_phases else ["mag", "phase"]

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
        reconstructed_specs = self.model.unpatchify(reconstructed_patches)
        
        #denorm if needed
        if self.model.config.norm_pix_loss:
            reconstructed_specs = self.model.denorm_patches(reconstructed_specs, model_output.input_specs)
        
        #convert the spectogram to the waveform
        reconstructed_waveforms = self.model.inverse_transform(reconstructed_specs, phases = model_output.phases) #shape of (batch_size, n_channels, sequence_len)

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

    
    def log_spectograms_and_waveforms(self, model_output: ViTMAEForEMG_PretrainingOutput):

        spectograms = self.reconstruct_spectograms(model_output)
        #log the waveforms
        waveforms = self.reconstruct_waveforms(model_output)

        if self.config.log_one_sample:
            self.logger.log_spectograms(*spectograms[0], plot_name="train_spectograms")
            self.logger.log_waveforms(*waveforms[0], plot_name="train_waveforms")
        else:
            specs_plot, names_plot = [], []
            for i, (s, n) in enumerate(spectograms):
                specs_plot += s
                names_plot += [name_plot + f"_{i}" for name_plot in n]
            self.logger.log_spectograms(specs_plot, names_plot, plot_name="train_spectograms")

            waveforms_plot, names_plot = [], []
            for i, (w, n) in enumerate(waveforms):
                waveforms_plot += w
                names_plot += [name_plot + f"_{i}" for name_plot in n]
            self.logger.log_waveforms(waveforms_plot, names_plot, plot_name="train_waveforms")
            
        

    def training_step(self, batch: Dict[str, torch.FloatTensor], batch_idx: int) -> torch.FloatTensor:
        
        out = self.model(batch["emg"])
        loss = out.loss
        self.log("train_loss", loss)
        if batch_idx % self.config.sample_log_interval_train == 0:
            self.log_spectograms_and_waveforms(out)
        return loss

    def validation_step(self, batch: Tuple[torch.FloatTensor, torch.FloatTensor], batch_idx: int) -> torch.FloatTensor:
        x, _ = batch
        out= self.model(x)
        loss = out.loss
        self.log("val_loss", loss)
        if batch_idx % self.config.sample_log_interval_val == 0:
            self.log_spectograms_and_waveforms(out)
        return loss

    #at the end of the epoch, log the metrics
    def on_train_epoch_end(self, outputs: List[torch.FloatTensor]) -> None:
        self.logger.dump_cache(prefix="train_", save_to_disk=True)

    def on_validation_epoch_end(self) -> None:
        self.logger.dump_cache(prefix="val_", save_to_disk=True)


    def configure_optimizers(self) -> dict[str, Any]:
        #this is slightly wrong but I am too tired to fix it
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, **self.config.scheduler_kwargs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
if __name__ == "__main__":

    example_emg = torch.randn(1, 10000)

    model = ViTMAEForEMG_Pretraining(ViTMAEForEMGConfig(sequence_len = example_emg.shape[-1],
                                                        losses = ["temporal_masked_only","spectral_masked_only"],
                                                        loss_weights = "equal"))

    output = model(example_emg)

    print(output.loss)
    print(output.loss_dict) 

