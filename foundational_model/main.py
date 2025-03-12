#add the parent directory to the path so we can import the foundational_model package
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from foundational_model.models import ViTMAE
from foundational_model.data import PretrainChannelWise_emg2qwerty
from foundational_model.utils.custom_logger import CustomPretrainLogger

import wandb
import torch
import yaml
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset

#seed the damn thing
pl.seed_everything(42)

#quick and dirty train script

data_path = "./data/89335547"

split_yaml = "./config/user/single_user.yaml"


#args for training and stuff, should be in a config file
window_length = 10000
batch_size = 32



#load the data
train_dataset = []
val_dataset = []
test_dataset = []

with open(split_yaml, "r") as f:
    split = yaml.safe_load(f)["dataset"]

    for entry in split["train"]:
        user = entry["user"]
        session = entry["session"]
        train_dataset.append(PretrainChannelWise_emg2qwerty(Path(os.path.join(data_path, f"{session}.hdf5")),
                                                            window_length = window_length))
        
        # break
    # print("working on val")
    
    #same for val and test
    for entry in split["val"]:
        user = entry["user"]
        session = entry["session"]
        val_dataset.append(PretrainChannelWise_emg2qwerty(Path(os.path.join(data_path, f"{session}.hdf5")),
                                                            window_length = window_length))
    for entry in split["test"]:
        user = entry["user"]
        session = entry["session"]
        test_dataset.append(PretrainChannelWise_emg2qwerty(Path(os.path.join(data_path, f"{session}.hdf5")),
                                                            window_length = window_length))
    

#concatenate the datasets
train_dataset = ConcatDataset(train_dataset)
val_dataset = ConcatDataset(val_dataset)
test_dataset = ConcatDataset(test_dataset)

#create the dataloaders
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 16)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = 16)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 16)


#instantiate the model
model_config = ViTMAE.LightningConfig(
    ViTMAE.ViTMAEForEMGConfig(
        sequence_len = window_length,
        n_fft =  64,
        hop_length = 16,
        log_spectogram = False,
        interpolation="linear",
        reshape_size = 256,
        patch_size = 8,
        bottleneck_dim = 2,
        # bottleneck_activation="ReLU",
        hidden_size = 16,
        intermediate_size = 64,
        num_attention_heads = 8,
        num_hidden_layers=6,

        decoder_hidden_size = 12,
        decoder_intermediate_size = 48,
        decoder_num_attention_heads = 12,
        decoder_num_hidden_layers=6,

        mask_ratio = 0.0,
        norm_pix_loss = True,
        losses = ["temporal_all", "spectral_all"],
        loss_weights=[0, 1],
        norm_method = "global",
    ),
    sample_log_interval=(int(len(train_loader)/4), int(len(val_loader)/4)),
    lr = 1e-3,
    lr_scheduler="CosineAnnealingWarmRestarts",
    lr_scheduler_kwargs = dict(T_0 = 25, T_mult = 2, eta_min = 1e-6),   
)

model = ViTMAE.ViTMAE_Pretraining_Lightning(model_config)

#instantiate the logger
wandb.init(project="emg2qwerty_pretraining")
logger = CustomPretrainLogger(project="emg2qwerty_pretraining",
                              save_dir = f"runs/{wandb.run.name}")
# checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_accuracy", mode="max")

#train
os.makedirs(logger.save_dir, exist_ok = True)
# save_path = os.path.join(logger.get_dump_dir(), "model_checkpoints")
# os.makedirs(save_path, exist_ok = True)
# print("saving to", save_path)
trainer = pl.Trainer(logger = logger, gpus = 1, max_epochs = 10)
trainer.fit(model, train_loader, val_loader)




                                    

        