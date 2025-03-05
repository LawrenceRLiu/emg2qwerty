#some custom callbacks

import torch
import wandb
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger


class SpectrumCallback(pl.Callback):
    
    