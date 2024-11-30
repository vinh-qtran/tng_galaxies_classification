import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

class DNN(nn.Module):
    def __init__(self,
                 input_dim : int,
                 hidden_dims : list,
                 output_dim : int,
                 dropout = 0.0,
                 activation = nn.ReLU(),
                 norm = nn.LayerNorm,
                 is_classification=False,):
        super(DNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.dropout = dropout

        self.activation = activation
        self.norm = norm
        self.is_classification = is_classification
        
        self.layers = []
        for i in range(len(hidden_dims)):
            self.layers.append(nn.Dropout(p=self.dropout))
            if i == 0:
                self.layers.append(self.norm(input_dim))
                self.layers.append(nn.Linear(input_dim, hidden_dims[i]))
            else:
                self.layers.append(self.norm(hidden_dims[i-1]))
                self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.layers.append(self.activation)

        self.layers.append(self.norm(hidden_dims[-1]))
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.last_activation = nn.Softmax(dim=1) if is_classification else nn.Identity()

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.last_activation(self.layers(x))