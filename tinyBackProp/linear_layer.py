"""
This script contains the concrete implementation of the Linear Layer
"""

import random
from typing import List

import numpy as np

from .param_layer import ParamLayer
# setting the seeds
np.random.seed(69)
random.seed(69)


class LinearLayer(ParamLayer):
    def __init__(self, in_features: int, out_features: int, weight_matrix: np.ndarray = None, bias: np.ndarray = None):
        super().__init__()
        if weight_matrix is not None and weight_matrix.shape != (out_features, in_features):
            raise ValueError(f"if the weight matrix is explicitly passed. The dimensions must match")

        self.in_features = in_features
        self.out_features = out_features
        self.weight = weight_matrix if weight_matrix is not None else np.random.randn(self.out_features, self.in_features)
        # self.bias = bias if bias is not None else np.random.rand(1, self.out_features) 

    def _verify_input(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        if x.shape[-1] != self.in_features:
            raise ValueError(f"The vector is expected to be a row vector of length {self.in_features}\nFound:{x.shape}")

        return x

    def forward(self, x: np.ndarray = None) -> np.ndarray:
        super().forward(x)
        x = self.last_x if x is None else x
        x = self._verify_input(x)
        return x @ self.weight.T  # + self.bias

    
    def param_grad(self, x: np.ndarray = None, upstream_grad: np.ndarray = None) -> np.ndarray:
        # make sure to save the last 'x'
        x = x if x is not None else self.last_x
        self.last_x = x        
        return upstream_grad.T @ x

    def grad(self, x: np.ndarray = None, upstream_grad: np.ndarray = None) -> np.ndarray:
        x = x if x is not None else self.last_x
        self.last_x = x        
        return upstream_grad @ self.weight

