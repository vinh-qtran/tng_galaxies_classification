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

import pickle

import data_loader
import inference
from models import mlp
from utils import data_reader, train_utils, splashback

import multiprocessing as mp
mp.set_start_method('fork')

###########################################################################################

data_path = '../../data/'

metadata = data_reader.Metadata(os.path.join(data_path,'metadata.pkl'))
clusters = data_reader.Clusters(os.path.join(data_path,'group_data/reduced_data.0.h5'))
galaxies = data_reader.Galaxies(os.path.join(data_path,'subhalo_data/reduced_data.0.h5'),
                                clusters,metadata.BoxSize/metadata.HubbleParam)

More15_splashback = splashback.More15Splashback()
def splashback_boundary_func(m200):
    rsp = More15_splashback.interp(m200)
    return (rsp, rsp)

loader = data_loader.GalaxyDataLoader(galaxies,
                                      boundary_func=splashback_boundary_func,
                                      trainval_ratio=0.8,
                                      train_ratio=0.9,)

train_loader, val_loader, test_loader = loader.get_data_loaders(batch_size=8192, num_workers=8)

for X,Y in train_loader:
    print('X:',X.shape)
    print('Y:',Y.shape)
    print('Y_mean:',Y.mean().item())
    break

############################################################################################

model = mlp.BaseMLP(input_dim = 6,
                    hidden_dims = [8,8,8],
                    output_dim = 1,
                    dropout = 0.0,
                    activation = nn.ReLU(),
                    norm = nn.BatchNorm1d,
                    last_activation = nn.Sigmoid())

trainer = train_utils.SupervisedTraining(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=200,
    lr=5e-5,
    criterion=nn.BCELoss(),
    optimizer=optim.Adam,
    scheduler=optim.lr_scheduler.CosineAnnealingLR,
    scheduler_params={'T_max': 50},
    is_classification=True,
    num_classes=2,
    device='mps',
)

outpath = 'training_result/mlp/splashback/'

trainer.train(save_training_stats_every=5, save_model_every=None, outpath=outpath)

##############################################################################################

model_state_dict = torch.load(outpath+'model/best.pth')['model_state_dict']

cluster_inference = inference.ClusterInference(test_loader,
                                               model,model_state_dict,
                                               mass_bins=np.logspace(3,5,16),
                                               radial_bins=np.linspace(0.5,2,16))

mass_bin_centers, radial_boundaries, radial_boundaries_err = cluster_inference.get_boundaries(sigmoid=True,n_MC=10000)

with open(outpath+'inference_result.pkl', 'wb') as f:
    pickle.dump({
        'mass_bin_centers': mass_bin_centers,
        'radial_boundaries': radial_boundaries,
        'radial_boundaries_err': radial_boundaries_err,
    }, f)