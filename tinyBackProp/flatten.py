"""
This script contains the code for the flatten layer of the library.
"""

import numpy as np
from .abstract_layer import Layer
from typing import Union


class FlattenLayer(Layer):
    def __init__(self) -> None:
        super().__init__()

    def _verify_input(self, x: np.ndarray) -> np.ndarray:
        # this function accepts a batch: 
        # Each element in the batch is expected to be at least 1 dimensional 
        if x.ndim  < 2:
            raise ValueError(f"The input is expected to be at least 2 dimensional")
        return x

    def forward(self, x: np.ndarray = None) -> np.ndarray:
        super().forward(x)
        x = self._verify_input(x)
        # extract the batch_size
        batch_size = x.shape[0]
        return np.reshape(x, (batch_size, -1))

    def grad(self, x: np.ndarray = None, upstream_grad: np.ndarray = None) -> np.ndarray:
        # set the default 'x' if needed
        x = self.last_x if x is None else x
        # the upstream gradient should have a shape that can be reshaped in the input shape
        return np.reshape(upstream_grad, x.shape)


