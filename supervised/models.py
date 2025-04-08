import numpy as np

import torch
import torch.nn as nn

class BaseMLP(nn.Module):
    def __init__(self,
                 input_dim : int,
                 hidden_dims : list,
                 output_dim : int = 1,
                 dropout = 0.0,
                 activation = nn.ReLU(),
                 norm = nn.BatchNorm1d,
                 last_activation = nn.Sigmoid()):
        super(BaseMLP, self).__init__()
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        layer_dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(layer_dims)-1):
            self.norms.append(norm(layer_dims[i]))

            self.layers.append(nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(layer_dims[i], layer_dims[i+1]),
                activation if i != len(layer_dims)-2 else last_activation,
            ))

    def forward(self, x):
        for norm, layer in zip(self.norms, self.layers):
            x = norm(x.view(-1, x.shape[-1])).view(x.shape)
            x = layer(x)
        return x