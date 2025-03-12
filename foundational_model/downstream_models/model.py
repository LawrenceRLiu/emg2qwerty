import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from hydra.utils import instantiate
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from typing import Any, ClassVar

from emg2qwerty.lightning import TDSConvCTCModule
from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.decoder import CTCGreedyDecoder, CTCBeamDecoder

from .utils import MLP
from .pool import MLP_Pool, AttentionPool
from .embedding import SimpleEmbedding
from ..models.ViTMAE import ViTMAE_Pretraining_Lightning


#using hydra because I am retarded
    
class DownstreamModel(nn.Module):
    N_HANDS:ClassVar[int] = 2
    N_CHANNELS:ClassVar[int] = 16
    def __init__(self, 
                    foundational_model_name:str,
                    foundational_model_checkpoint:str,
                    freeze_foundational_model:bool,
                    output_size:int,
                    pre_pooling_mlp_config:DictConfig,
                    post_pooling_mlp_config:DictConfig,
                    pooling_config: DictConfig,
                    embedding_config: DictConfig):
        super().__init__()
        print("here")

        #load the foundational model
        print(f"loading foundational model {foundational_model_name} from {foundational_model_checkpoint}")
        self.load_foundational_model(foundational_model_name, foundational_model_checkpoint, freeze_foundational_model)

        print("encoder output size", self.foundational_model.get_encoder_output_size())
        self.embedding = instantiate(embedding_config, embedding_size = self.foundational_model.get_encoder_output_size(),
                                     n_channels = (self.N_CHANNELS)//self.foundational_model.config.input_num_channels)
        self.pre_pooling_mlp = MLP(input_size = self.foundational_model.get_encoder_output_size(),
                                   final_activation = True,
                                           **pre_pooling_mlp_config)
        # raise ValueError("stop here")
        hidden_size = self.pre_pooling_mlp.output_size
        print()
        print("pooling config", pooling_config)
        self.pool = instantiate(pooling_config, embedding_size = hidden_size)
        self.post_pooling_mlp = MLP(
                                            input_size = hidden_size,
                                            output_size = output_size,
                                            **post_pooling_mlp_config)
        
        
        self.final_activation = nn.LogSoftmax(dim=-1)


    def load_foundational_model(self, foundational_model_name:str, foundational_model_checkpoint:str, freeze_foundational_model:bool):
        
        if foundational_model_name == "ViTMAE":
            foundational_model_pl = ViTMAE_Pretraining_Lightning.load_from_checkpoint(foundational_model_checkpoint)
        else:
            #TODO: add your foundational model here
            raise ValueError(f"foundational model {foundational_model_name} not supported")
        
        self.foundational_model = foundational_model_pl.model

        #set to encoder only mode
        # self.foundational_model.encoder_only(freeze_foundational_model)

    def forward(self, emg_data):

        #emg data of shape (batch_size, 2, channels, sequence_len)
        #reshape to the shape expected by the foundational model
        print("emg_data", emg_data.shape)
        emg_data = emg_data.permute(1,2,3,0)
        n_batch, _, n_channels, sequence_len = emg_data.shape
        assert sequence_len == self.foundational_model.config.sequence_len, f"sequence length must match the foundational model sequence length: {sequence_len} != {self.foundational_model.config.sequence_len}"
        x = self.foundational_model(emg_data.view(-1, self.foundational_model.config.input_num_channels, sequence_len))
        print("x", x.shape)
        #reshape to the shape expected by the DownstreamHead
        x = x.view(n_batch, 2, -1, x.shape[-1])

        #x is expected to be of shape (batch_size, 2, channels, embedding_size)

        #add the positional embedding
        x = self.embedding(x)

        #pass through the mlp
        x = self.pre_pooling_mlp(x)

        print("pre_pooling_mlp", x.shape)

        #pool across the channels
        x = self.pool(x) #shape (batch_size, embedding_size)

        #pass through the mlp
        x = self.post_pooling_mlp(x)

        #apply the final activation
        x = self.final_activation(x)

        print("final_x", x.shape)
        return x


    
class DownstreamModelLighting(pl.LightningModule):
    """Downstream task model, built off the TDSConvCTCModule"""

    def __init__(self, 
                foundational_model_name:str,
                foundational_model_checkpoint:str,
                DownstreamHead_config:DictConfig,
                # pooling_config: DictConfig,
                # embedding_config: DictConfig,
                optimizer: DictConfig,
                lr_scheduler: DictConfig,
                decoder: DictConfig):
        super().__init__()
        print("optimizer", optimizer)
        print("lr_scheduler", lr_scheduler)
        # raise ValueError("stop here")  
        self.save_hyperparameters()
        print("self.hparams", type(self.hparams), self.hparams.keys())
        # raise ValueError
        # self.hparams.optimizer = optimizer
        # self.hparams.lr_scheduler = lr_scheduler
        print("here")
        print("DownstreamHead_config", DownstreamHead_config)
        #load the foundational model
        self.model = DownstreamModel(
            foundational_model_name = foundational_model_name,
            foundational_model_checkpoint = foundational_model_checkpoint,
            output_size = charset().num_classes,
            **DownstreamHead_config
        )
        
        # instantiate(
        #     {"_target_":DownstreamModel},
        #     foundational_model_name = foundational_model_name,
        #     foundational_model_checkpoint = foundational_model_checkpoint,
        #     output_size = charset().num_classes,
        #     # pooling_config = pooling_config,
        #     # embedding_config = embedding_config,
        #     _recursive_=False,
        #     **DownstreamHead_config
        # )


        # Decoder
        self.decoder = instantiate(decoder)
        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


        

    