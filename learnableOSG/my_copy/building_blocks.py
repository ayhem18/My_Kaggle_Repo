""" This script contains functionalities used across the different scripts in the folder
"""

import torch
import numpy as np
import random 
from torch import nn
from typing import Iterator, Tuple, Union, List
from pathlib import Path

from torch.nn.modules.module import Module

# let's first set the random seeds 
random.seed(69)
np.random.seed(69)
torch.manual_seed(69)


class LinearBlock(nn.Module):
    # let's define a class method that will map the activation name to the correct layer
    activation_functions = {"leaky_relu": nn.LeakyReLU,  
                            "relu": nn.ReLU, 
                            "tanh": nn.Tanh}
    
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 dropout_rate: float = 0,
                 activation: str = 'leaky_relu',
                 batch_normalization: bool = True,
                 is_final: bool = False,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # the idea here is quite simple. 
        components = [nn.Linear(in_features=in_features, out_features=out_features)]

        if dropout_rate > 0:
            # dropout rate of '0' means the user is not interested in using dropout at this point. Thus, the layer
            # the layer should not be created as it will be a usual linear layer
            components.append(nn.Dropout(p=dropout_rate))

        # depending on the value of 'is_final' 
        if not is_final:
            norm_layer = nn.BatchNorm1d(num_features=out_features)
            activation_layer = self.activation_functions[activation]()
            # make sure to consider the 'batch_normalization' argument
            components.extend(([norm_layer, activation_layer] if batch_normalization else [activation_layer]))
        
        self._block = nn.Sequential(*components)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._block.forward(x)
    
    def __str__(self) -> str:
        return self._block.__str__()
    
    @property
    def block(self):
        return self._block
    
    @block.setter
    def block(self, new_block: Union[nn.Sequential, nn.Module]):
        # make sure the new block is either 
        if isinstance(new_block, (nn.Sequential, nn.Module)):
            raise TypeError((f"The block is expected to be either of type {nn.Module} or {nn.Sequential}\n"
                             f"Found: {type(new_block)}"))
    
    def children(self) -> Iterator[Module]:
        return self.block.children()

    def named_children(self) -> Iterator[Module]:
        return self.block.children()
    

class LinearNet(nn.Module):
    def build(self, 
            hidden_units: List[int],   
            activation: str,
            dropout_rate: Union[float, List[float]],
            batch_normalization: Union[bool, List[bool]] = True,
            last_layer_is_final: bool = True):

        # make sure the length of 'hidden units' argument is at least 2
        if len(hidden_units) < 2:
            raise ValueError(f"At least 2 hidden units must be specified.\nFound only {len(hidden_units)}") 

        # convert the batch_normalization to a list
        batch_normalization = batch_normalization if isinstance(batch_normalization, List) else [batch_normalization for _ in hidden_units[:-1]]

        if len(batch_normalization) != len(hidden_units) - 1:
            raise ValueError(f"Too many batch norm arguments were given. Excpted: {len(hidden_units) - 1}\nFound: {len(batch_normalization)}")
        

        # convert the dropout argument to a working format
        dropout_rate = dropout_rate if isinstance(dropout_rate, List) else [0 for _ in (hidden_units[:-2])] + [dropout_rate] 

        if len(batch_normalization) != len(hidden_units) - 1:
            raise ValueError(f"Too many dropout arguments were given. Excpted: {len(hidden_units) - 1}\nFound: {len(dropout_rate)}")

        # build the feed forward neural network 
        layers = [LinearBlock(in_features=hidden_units[i], 
                              out_features=hidden_units[i + 1], 
                              activation=activation, 
                              batch_normalization=batch_normalization[i],
                              dropout_rate=dropout_rate[i], 
                              is_final=False) for i in range(len(hidden_units) - 2)
                              ]

        layers.append(LinearBlock(in_features=hidden_units[-2], 
                                  out_features=hidden_units[-1], 
                                  is_final=last_layer_is_final, 
                                  activation=activation, 
                                  dropout_rate=dropout_rate[-1],
                                  batch_normalization=batch_normalization[-1]))

        return nn.Sequential(*layers)
    
    def __init__(self, 
            hidden_units: List[int],   
            activation: str,
            dropout_rate: Union[float, List[float]],
            batch_normalization: Union[bool, List[bool]] = True,
            last_layer_is_final: bool = True):
        
        super().__init__()
        # let's make it a simple encoder
        self.net = self.build(hidden_units=hidden_units, 
                              dropout_rate=dropout_rate, 
                              batch_normalization=batch_normalization,
                              activation=activation, 
                              last_layer_is_final=last_layer_is_final)

    def forward(self, x: torch.Tensor):
        return self.net.forward(x)

    def __str__(self) -> str:
        return self.net.__str__()
    
    def children(self) -> Iterator[Module]:
        return self.net.children()

    def named_children(self) -> Iterator[Module]:
        return self.net.children()
