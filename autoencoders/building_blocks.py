# first let's build a simple linear block
import torch
import numpy as np
from torch import nn
import random 


import os, sys
from pathlib import Path

HOME = os.getcwd()
DATA_FOLDER = os.path.join(HOME, 'data') 
current = HOME



while 'pytorch_modular' not in os.listdir(current):
    current = Path(current).parent

sys.path.append(str(current))
sys.path.append(os.path.join(current, 'pytorch_modular'))

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
                 activation: str = 'leaky_relu',
                 is_final: bool = True,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # the idea here is quite simple. 
        components = [nn.Linear(in_features=in_features, out_features=out_features)]
        # depending on the value of 'is_final' 
        if not is_final:
            norm_layer = nn.BatchNorm1d(num_features=out_features)
            activation_layer = self.activation_functions[activation]()
            components.extend([norm_layer, activation_layer])
        
        self.block = nn.Sequential(*components)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block.forward(x)
    
    def __str__(self) -> str:
        return self.block.__str__()
    

class LinearNet(nn.Module):
    def build(self, 
            in_features: int, 
            out_features: int, 
            num_layers: int, 
            activation: str):
        # the first and final layer are always present
        num_layers = num_layers + 2 
        # compute the number of hidden units
        hidden_units = [int(u) for u in np.linspace(in_features, out_features, num=num_layers)] 

        layers = [LinearBlock(in_features=hidden_units[i], out_features=hidden_units[i + 1], activation=activation, is_final=False) for i in range(len(hidden_units) - 2)]

        # the last linear block should be set as 'final'
        layers.append(LinearBlock(in_features=hidden_units[-2], out_features=hidden_units[-1], is_final=True))

        return nn.Sequential(*layers)
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 num_layers: int = 1,
                 activation: str = 'leaky_relu', 
                 *args, 
                 **kwargs):
        
        super().__init__(*args, **kwargs)
        # let's make it a simple encoder
        self.net = self.build(in_features=in_features, out_features=out_features, num_layers=num_layers, activation=activation)

    def forward(self, x: torch.Tensor):
        return self.net.forward(x)

    def __str__(self) -> str:
        return self.net.__str__()
    




class UnderCompleteAE(nn.Sequential):
    def __init__(self, in_features: int, 
                 bottleneck: int = None,
                 num_layers: int = 2, 
                 activation: str = 'leaky_relu', 
                 *args, 
                 **kwargs):

        super().__init__(*args, **kwargs)

        if bottleneck is None:
            bottleneck = in_features // 2
        
        # The idea is simple, let's first build
        self._encoder = LinearNet(in_features=in_features, 
                                 out_features=bottleneck, 
                                 num_layers=num_layers,
                                 activation=activation)
        
        self._decoder = LinearNet(in_features=bottleneck, 
                                 out_features=in_features, 
                                 num_layers=num_layers, 
                                 activation=activation)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self._decoder(self._encoder(x))

    def encode(self, x:torch.Tensor):
        return self._encoder(x)

    def decode(self, x:torch.Tensor):
        return self._decoder(x)
    