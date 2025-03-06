import yaml
from foundational_model.models import ViTMAE
from foundational_model.data import PretrainChannelWise_emg2qwerty
from foundational_model.utils.custom_logger import CustomPretrainLogger

import wandb
import os
import torch
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
#quick and dirty train script

data_path = "./data/89335547"

split_yaml = "/home/lawrence/emg2qwerty/config/user/single_user.yaml"


#args for training and stuff, should be in a config file
window_length = 5000
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
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 8)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = 8)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 8)


#instantiate the model
model_config = ViTMAE.LightningConfig(
    ViTMAE.ViTMAEForEMGConfig(
        sequence_len = window_length,
        n_fft =  128,
        hop_length = 32,
        log_spectogram = False,
        hidden_size = 128,
        intermediate_size = 1024,
        num_attention_heads = 8,

        decoder_hidden_size = 128,
        decoder_intermediate_size = 1024,
        decoder_num_attention_heads = 8,

        mask_ratio = 0.5,
        norm_pix_loss = True,
        # losses = ["temporal_masked_only", "spectral_masked_only"],
    ),
    sample_log_interval=(int(len(train_loader)/4), int(len(val_loader)/4)),
)

model = ViTMAE.ViTMAE_Pretraining_Lightning(model_config)

#instantiate the logger
wandb.init(project="emg2qwerty_pretraining")
logger = CustomPretrainLogger(project="emg2qwerty_pretraining",
                              save_dir = "logs")

#train
trainer = pl.Trainer(logger = logger, gpus = 1, max_epochs = 100)
trainer.fit(model, train_loader, val_loader)




                                    

        