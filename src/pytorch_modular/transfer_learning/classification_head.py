"""
This script contains different general classes of classification heads with different design choices: either my own
design or inspired from other sources.
"""


import torch
import numpy as np
import torch.nn.functional as f
from torch import nn


class ExponentialClassifierHead(nn.Module):
    def __init__(self, num_classes: int,
                 num_layers: int,
                 in_features: int):
        # as usual call the super class constructor
        super().__init__()
        # the shape used in the classifier's output
        self.output = num_classes if num_classes > 2 else 1
        self.num_layers = num_layers
        self.in_features = in_features
        self._build_classifier()

    def _build_classifier(self):
        base_power = int(np.log2(self.in_features))
        powers = np.linspace(start=int(np.log2(self.output)), stop=base_power, num=self.num_layers)
        # make sure to convert to integers
        num_units = [int(2 ** p) for p in powers][::-1]
        # set the last element to the actual number of classes
        num_units[-1] = self.output
        num_units = [self.features] + num_units

        layers = [nn.Linear(in_features=num_u, out_features=num_units[i + 1]) for i, num_u in enumerate(num_units[:-1])]
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = f.relu(layer(x))

        return self.layers[-1](x)
