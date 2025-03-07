#!/bin/bash

conda env create -f environment.yml
conda activate emg2qwerty


#download all data, uncomment if you want to download all data
# wget https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz
# tar -xvzf emg2qwerty-data-2021-08.tar.gz
# ln -s ./emg2qwerty-data-2021-08 ./emg2qwerty/data

#download single user data
mkdir -p ./data/89335547
#manually download the data and unzip it to ./data/89335547
# https://ucla.box.com/s/3xc4nwpfjfpo6ydjs94t0v2kuq37d5eg



#additional packages
conda install -y conda-forge::transformers conda-forge::wandb
pip install protobuf==3.20.*
conda install -y lightning -c conda-forge