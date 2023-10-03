"""
This script contains my personnel implementation of the most popular layers in Deep Learning
starting with the Linear Layer . 
"""
import warnings

import numpy as np

from abc import ABC, abstractmethod


@abstractmethod
class Layer(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def gradient(self) -> np.ndarray:
        pass

    @abstractmethod
    def initialize(self):
        pass


class LinearLayer(Layer):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 weights: np.ndarray = None,
                 bias: np.ndarray = None):

        self.in_features = in_features
        self.out_features = out_features
        # the parameters
        self.weights = None
        self.bias = None

        if weights is not None:
            # make sure the dimensions are: 'in_features' 'out_features'
            if weights.shape != (self.in_features, self.out_features):
                raise ValueError(f"The given weights do not match the given dimensions\n"
                                 f"Expected: {(self.in_features, self.out_features)}\n"
                                 f"Found: {weights.shape}")
            self.weights = weights

        if bias is not None:
            bias = np.expand_dims(bias, axis=-1) if len(bias.shape) == 1 else bias
            # if the bias vector is given as a column vector, transpose it
            if len(bias.shape) == 2 and bias.shape[-1] == 1:
                bias = np.transpose(bias)
            self.bias = bias

        # initialize any of the initialized components
        w, b = self.initialize()
        self.weights, self.bias = (w if self.weights is None else self.weights,
                                   b if self.bias is None else self.bias)

    def initialize(self):
        # this function is called to randomly initialize weights
        weights = np.random.rand(self.in_features, self.out_features)
        bias = np.random.rand(1, self.out_features)
        return weights, bias

    def forward(self, x: np.ndarray):
        # un-batched input
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=-1)

        if len(x.shape) == 2 and x.shape[0] == self.in_features:
            # transpose the input
            # make sure to let the user know
            x = np.transpose(x)
            warnings.warn(f"The input is given as column vectors. We are expecting row vectors\n"
                          f"The input was transposed internally to the shape {x.shape[0], self.in_features} ")

        # the input should be of shape (None, self.in_features)
        if len(x.shape) != 2 or x.shape[1] != self.in_features:
            raise ValueError(f"The layer expects a 2-d input of row vectors with length: {self.in_features}\n"
                             f"Found: vectors of length: {x.shape[1]}")

        return self.weights @ x + self.bias

    def gradient(self) -> np.ndarray:
        pass

