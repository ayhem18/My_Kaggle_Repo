""" 
This script contains functionalities to build networks out of the simple 
"""

from typing import List
import numpy as np

from copy import deepcopy
from tinyBackProp.abstract_layer import Layer
from tinyBackProp.linear_layer import ParamLayer


class Network:
    def __init__(self, layers: List[Layer]) -> None:
        self.layers = deepcopy(layers)

    def forward(self, x: np.ndarray):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x

    def backward(self, loss_grad: np.ndarray, learning_rate: float):

        upstream_grad = loss_grad

        grads = [upstream_grad]

        for i in range(len(self.layers) - 1, -1, -1):
            param_grad = None

            # calculate the upstream gradient
            new_upstream_grad = self.layers[i].grad(upstream_grad=upstream_grad)

            # first calculate the gradient that might be used to update the parameters of a model
            if isinstance(self.layers[i], ParamLayer):
                param_grad = self.layers[i].param_grad(upstream_grad=upstream_grad)
                # update teh gradient
                self.layers[i].update(param_grad, learning_rate)
                grads.append(param_grad)
            
            upstream_grad = new_upstream_grad.copy()

        return grads
