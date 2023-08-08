"""
This script contains different general classes of classification heads with different design choices: either my own
design or inspired from other sources.
"""

import torch
import numpy as np
import torch.nn.functional as f

from torch import nn
from collections import OrderedDict
from typing import List, Sequence, Iterator


# TODO: CONCEIVE A METHOD TO THOROUGHLY TEST THE `ExponentialClassifier`
class ExponentialClassifier(nn.Module):
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


class GenericClassifier(nn.Module):
    """
    This is a generic classifier where the architecture is fully defined by the input.
    The user simply indicates the initial_input, and the number of classes, as well as the number of hidden units.
    """

    def _build_classifier(self) -> nn.Sequential:
        modules = OrderedDict()

        in_features = self._in_features
        for i, hu in enumerate(self.hidden_units, 1):
            modules[f'layer_{i}'] = nn.Linear(in_features=in_features,
                                              out_features=hu)
            # make sure to update the in_features variable
            in_features = hu
            # make sure to add a non-linearity layer
            modules[f'relu_{i}'] = nn.ReLU()

        # set the last layer: the output layer
        modules[f'layer_{len(self.hidden_units) + 1}'] = nn.Linear(in_features=in_features,
                                                                   out_features=self._num_classes)

        self.classifier = nn.Sequential(modules)

    def __init__(self,
                 in_features: int,
                 num_classes: int,
                 hidden_units: Sequence[int],
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        if num_classes < 2:
            raise ValueError('The number of classes cannot be less than 2.\n'
                             f'FOUND: {num_classes}')

        self._in_features = in_features
        self._num_classes = 1 if num_classes == 2 else num_classes
        self.hidden_units = hidden_units
        self.classier = None

        self._build_classifier()

    # the num_classes and in_features fields affect the entire architecture of the classifier: They should
    # have specific setter methods that modify the `self.classifier` field
    @property
    def num_classes(self):
        return self._num_classes

    # a setter for num_classes and in_features
    @num_classes.setter
    def num_classes(self, x: int):
        self._num_classes = x if x > 2 else 1
        self._build_classifier()

    @property
    def in_features(self):
        return self._in_features

    @in_features.setter
    def in_features(self, x: int):
        self._in_features = x
        self._build_classifier()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier.forward(x)

    def children(self) -> Iterator[nn.Module]:
        return self.classifier.children()

    def named_children(self) -> Iterator[nn.Module]:
        return self.classifier.named_children()

    def modules(self) -> Iterator[nn.Module]:
        return self.classifier.modules()
