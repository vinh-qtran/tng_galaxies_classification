import os
import sys
import subprocess
from pathlib import Path

repo_root = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
).stdout.strip()

sys.path.append(repo_root)

##########################################################################################

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import data_loader
from models import mlp
from utils import train_utils

mp.set_start_method('fork')

###########################################################################################

import importlib

importlib.reload(data_loader)

###########################################################################################

data_path = '../../data/'

metadata = data_loader.Metadata(os.path.join(data_path,'metadata.pkl'))
clusters = data_loader.Clusters(os.path.join(data_path,'group_data/reduced_data.0.h5'))
galaxies = data_loader.Galaxies(os.path.join(data_path,'subhalo_data/reduced_data.0.h5'),
                                clusters,metadata.BoxSize/metadata.HubbleParam)

loader = data_loader.GalaxyDataLoader(galaxies,
                                      boundaries=(1,1),
                                      trainval_ratio=0.9,
                                      train_ratio=0.8,)

train_loader, val_loader, test_loader = loader.get_loaders(batch_size=2048, num_workers=8)

