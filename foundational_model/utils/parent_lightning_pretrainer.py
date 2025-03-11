import pytorch_lightning as pl
from transformers.configuration_utils import PretrainedConfig
from typing import Any, Union, Dict, Tuple, List, Literal, Optional
import torch.nn as nn
import torch
from dataclasses import dataclass, InitVar, field
from .custom_logger import CustomPretrainLogger

#parent class for the lightning module for pretraining
@dataclass
class LightningConfig:
    model_config: PretrainedConfig
    optimizer: Union[torch.optim.Optimizer, str] = "Adam"
    lr: float = 1e-3
    lr_scheduler: Union[torch.optim.lr_scheduler._LRScheduler, str] = "none"
    lr_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    sample_log_interval: InitVar[Union[int, Tuple[int,int]]] = 100,  #tuple allows us to pick different logging intervals for train and val
    log_one_sample: bool = True, #log one sample of the input and output spectograms or all the samples in a batch,

    def __post_init__(self, sample_log_interval):
        print("sample_log_interval", sample_log_interval)
        if isinstance(sample_log_interval, int):
            self.sample_log_interval_train = sample_log_interval
            self.sample_log_interval_val = sample_log_interval
        else:
            self.sample_log_interval_train, self.sample_log_interval_val = sample_log_interval
        


@dataclass(kw_only=True)
class FoundationalModelOutput:
    """Additional output from the foundational model

    losses:dict[str, tuple[torch.FloatTensor, float]]: the name, loss and weight of the loss
    """
    loss: torch.FloatTensor #the loss value to backpropagate
    losses:Dict[str, Tuple[float, float]] #decomposition into individual losses and their weights
    input_waveforms: torch.FloatTensor #the input waveform
    input_specs: torch.FloatTensor #the input spectrogram
    phases: torch.FloatTensor #the phases of the input spectrogram

class ParentModel(nn.Module):

    #the functions we expect the model to have
    def forward(self, x: torch.FloatTensor) -> FoundationalModelOutput:
        raise NotImplementedError
    
    def reconstruct_spectograms(self, model_output: FoundationalModelOutput) -> List[Tuple[torch.FloatTensor, str]]:
        """Reconstruct the spectograms from the model output
        Args:
            model_output (ViTMAEForEMG_PretrainingOutput): the model output
        Returns:
            List[Tuple[List[torch.FloatTensor], List[str]]]: the reconstructed spectograms and their names
            return a seperate Tuple for entry in the batch, as they will be plotted seperately
        """
        raise NotImplementedError
    
    def reconstruct_waveforms(self, model_output: FoundationalModelOutput) -> List[Tuple[torch.FloatTensor, str]]:
        """Reconstruct the waveforms from the model output
        Args:
            model_output (ViTMAEForEMG_PretrainingOutput): the model output
        Returns:
            List[Tuple[List[torch.FloatTensor], List[str]]]: the reconstructed waveforms and their names
            return a seperate Tuple for entry in the batch, as they will be plotted seperately
        """
        raise NotImplementedError
    

# parent class for the lightning module for pretraining
class Pretraining_Lightning(pl.LightningModule):
    model: ParentModel
    logger: CustomPretrainLogger

    def __init__(self, LightningConfig: LightningConfig):
                 
        super().__init__()
        self.save_hyperparameters()

        self.config = LightningConfig

    def log_spectograms_and_waveforms(self, model_output: FoundationalModelOutput,
                                      mode: Literal["train","val"] = "train") -> None:

        spectograms = self.model.reconstruct_spectograms(model_output)
        #log the waveforms
        waveforms = self.model.reconstruct_waveforms(model_output)


        if self.config.log_one_sample:
            self.logger.log_spectograms(*spectograms[0], plot_name=f"{mode}_spectograms")
            self.logger.log_waveforms(*waveforms[0], plot_name=f"{mode}_waveforms")
        else:
            raise NotImplementedError("Logging all samples in a batch is too memory intensive rn")
            specs_plot, names_plot = [], []
            for i, (s, n) in enumerate(spectograms):
                specs_plot += s
                names_plot += [name_plot + f"_{i}" for name_plot in n]
            self.logger.log_spectograms(specs_plot, names_plot, plot_name=f"{mode}_spectograms")

            waveforms_plot, names_plot = [], []
            for i, (w, n) in enumerate(waveforms):
                waveforms_plot += w
                names_plot += [name_plot + f"_{i}" for name_plot in n]
            self.logger.log_waveforms(waveforms_plot, names_plot, plot_name=f"{mode}_waveforms")

    def forward(self, x: torch.FloatTensor) -> Any:
        return self.model(x)
    
    def training_step(self, batch: Dict[str, torch.FloatTensor], batch_idx: int) -> torch.FloatTensor:
        
        out = self.model(batch["emg"])
        loss = out.loss
        self.log("train_loss", loss)
        for key, (loss_val, weight) in out.losses.items():
            self.log(f"train_{key}", loss_val)
            self.log(f"train_{key}_weight", weight)
        if batch_idx % self.config.sample_log_interval_train == 0:
            self.log_spectograms_and_waveforms(out, mode=f"train_{batch_idx}")
        return loss

    def validation_step(self, batch: Tuple[torch.FloatTensor, torch.FloatTensor], batch_idx: int) -> torch.FloatTensor:
        x = batch["emg"]
        out= self.model(x)
        loss = out.loss
        self.log("val_loss", loss)
        for key, (loss_val, weight) in out.losses.items():
            self.log(f"val_{key}", loss_val)
            self.log(f"val_{key}_weight", weight)

        if batch_idx % self.config.sample_log_interval_val == 0:
            self.log_spectograms_and_waveforms(out, mode = f"val_{batch_idx}")
        return loss
    
    #at the end of the epoch, log the metrics
    def on_train_epoch_end(self) -> None:
        self.logger.dump_cache(prefix=f"train_plots/{self.current_epoch}", save_to_disk=True,
                                 dump_matched_only= True, matching_substr = f"train")

    def on_validation_epoch_end(self) -> None:
        self.logger.dump_cache(prefix=f"val_plots/{self.current_epoch}", save_to_disk=True,
                               dump_matched_only= True, matching_substr = f"val")
    
    def configure_optimizers(self) -> dict[str, Any]:
        if isinstance(self.config.optimizer, str):
            #instantiate the optimizer
            optimizer = eval(f"torch.optim.{self.config.optimizer}")(self.parameters(), lr=self.config.lr)
            #very unsafe way to do it, fix it if you have a problem with it
        else:
            optimizer = self.config.optimizer(self.parameters(), lr=self.config.lr)

        if isinstance(self.config.lr_scheduler, str):
            if self.config.lr_scheduler.lower() == "none":
                return {"optimizer": optimizer}
            scheduler = eval(f"torch.optim.lr_scheduler.{self.config.lr_scheduler}")(optimizer, **self.config.lr_scheduler_kwargs)
        else:
            scheduler = self.config.lr_scheduler(optimizer, **self.config.lr_scheduler_kwargs)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}