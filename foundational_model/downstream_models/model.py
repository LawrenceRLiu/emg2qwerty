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


class DownstreamHead(nn.Module):
    """DownstreamHead for the downstream task"""

    def __init__(self, 
                 input_size:int,
                 n_channels:int,
                 output_size:int,
                 pre_pooling_mlp_config:DictConfig,
                 post_pooling_mlp_config:DictConfig,
                 pooling_config: DictConfig,
                embedding_config: DictConfig):
        """DownstreamHead for the downstream task"""

        super().__init__()
        self.input_size = input_size

        self.embedding = instantiate(embedding_config, embedding_size = input_size,
                                     n_channels = n_channels//2)
        self.pre_pooling_mlp = instantiate({"__target__":MLP}, input_size = input_size,
                                           **pre_pooling_mlp_config)

        hidden_size = self.pre_pooling_mlp.output_size
        print()
        print("pooling config", pooling_config)
        self.pooling = instantiate(pooling_config, embedding_size = hidden_size)
        self.post_pooling_mlp = instantiate({"__target__":MLP},
                                            input_size = hidden_size,
                                            output_size = output_size,
                                            **post_pooling_mlp_config)
        
        
        self.final_activation = nn.LogSoftmax(dim=-1)
    
    def forward(self, x):   
        #x is expected to be of shape (batch_size, 2, channels, embedding_size)

        #add the positional embedding
        x = self.embedding(x)

        #pool across the channels
        x = self.pool(x) #shape (batch_size, embedding_size)

        #pass through the mlp
        x = self.mlp(x)

        #apply the final activation
        x = self.final_activation(x)

        return x
    
class DownstreamModel(nn.Module):
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
        self.DownstreamHead = DownstreamHead(input_size = self.foundational_model.get_encoder_output_size(),
                                             n_channels = 32 // self.foundational_model.config.input_num_channels,
                                                output_size = output_size,
                                                pre_pooling_mlp_config = pre_pooling_mlp_config,
                                                post_pooling_mlp_config = post_pooling_mlp_config,
                                                pooling_config = pooling_config,
                                                embedding_config = embedding_config)
        

    def load_foundational_model(self, foundational_model_name:str, foundational_model_checkpoint:str, freeze_foundational_model:bool):
        
        if foundational_model_name == "ViTMAE":
            foundational_model_pl = ViTMAE_Pretraining_Lightning.load_from_checkpoint(foundational_model_checkpoint)
        else:
            #TODO: add your foundational model here
            raise ValueError(f"foundational model {foundational_model_name} not supported")
        
        self.foundational_model = foundational_model_pl.model

        #set to encoder only mode
        self.foundational_model.encoder_only(freeze_foundational_model)

    def forward(self, emg_data):

        #emg data of shape (batch_size, 2, channels, sequence_len)
        #reshape to the shape expected by the foundational model

        n_batch, _, n_channels, sequence_len = emg_data.shape
        assert sequence_len == self.foundational_model.config.sequence_len, f"sequence length must match the foundational model sequence length: {sequence_len} != {self.foundational_model.config.sequence_len}"
        x = self.foundational_model(emg_data.view(-1, self.foundational_model.config.input_num_channels, sequence_len))

        #reshape to the shape expected by the DownstreamHead
        x = x.view(n_batch, -1, x.shape[-1])

        return self.DownstreamHead(x)

    
class DownstreamModelLighting(TDSConvCTCModule, pl.LightningModule):
    """Downstream task model, built off the TDSConvCTCModule"""

    def __init__(self, 
                foundational_model_name:str,
                foundational_model_checkpoint:str,
                DownstreamHead_config:DictConfig,
                optimizer: DictConfig,
                lr_scheduler: DictConfig,
                decoder: DictConfig):

        pl.LightningModule.__init__(self) 
        print("optimizer", optimizer)
        print("lr_scheduler", lr_scheduler)
        # raise ValueError("stop here")  
        self.save_hyperparameters()
        self.hparams.optimizer = optimizer
        self.hparams.lr_scheduler = lr_scheduler

        #load the foundational model
        self.model = instantiate(
            DownstreamHead_config,
            foundational_model_name = foundational_model_name,
            foundational_model_checkpoint = foundational_model_checkpoint,
            output_size = charset().num_classes,
            _recursive_ = False,
            # **DownstreamHead_config
        )


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

    def configure_optimizers(self) -> dict[str, Any]:
        print("self.hparams", self.hparams)
        return utils.instantiate_optimizer_and_scheduler(
            self.model.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )
    
        

        

    