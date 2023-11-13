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

    # def local_param_grads(self, x: np.ndarray) -> List[np.ndarray]:
    #     batch_size, x_m = x.shape

    #     def gradient_matrix(i, j):
    #         # the gradient matrix should have 0s in all columns different from j
    #         result = np.zeros(shape=(batch_size, self.out_features), dtype=np.float32)

    #         for k in range(batch_size):
    #             result[k, j] = x[k, i]

    #         return result

    #     grads = [gradient_matrix(i, j) for i in range(self.in_features) for j in range(self.out_features)]
    #     return grads

    # def _reshape_grad(self, local_grads: list[float]) -> np.ndarray:
    #     grad_reshaped = np.reshape(np.asarray(local_grads), self.weight.shape)
    #     return grad_reshaped

    # def local_x_grad(self, x: np.ndarray):
    #     x = self.last_x if x is None else x

    #     if x is None:
    #         raise TypeError(f"the method is expecting non None input")

    #     self.last_x = x

    #     # compute the gradient of the output with respect to a given x_i
    #     batch_size, x_m = x.shape

    #     def gradient_matrix(i, j):
    #         # the gradient matrix should have 0s in all columns different from j
    #         result = np.zeros(shape=(batch_size, self.out_features), dtype=np.float32)

    #         for k in range(self.out_features):
    #             result[i, k] = self.weight[j, k]

    #         return result

    #     grads = [gradient_matrix(i, j) for i in range(x.shape[0]) for j in range(x.shape[1])]
    #     return grads

    # def grad(self, x: np.ndarray = None, upstream_grad: np.ndarray = None) -> np.ndarray:
    #     x = self.last_x if x is None else x
    #
    #     if x is None:
    #         raise TypeError(f"the method is expecting non-None input")
    #
    #     self.last_x = x
    #
    #     # first step generate the local gradient
    #     local_grads = self.local_x_grad(x)
    #     # make sure the shape of the upstream_gram matches that of any given local grad
    #     if local_grads[0].shape != upstream_grad.shape:
    #         raise ValueError((f"The upstream gradient is expected to be of the same shape as the local gradients\n "
    #                           f"Expected: {local_grads[0].shape}. Found: {upstream_grad.shape}"))
    #
    #     # the next step is to compute the upstream gradient with respect to each parameter
    #     final_grads = [np.sum(lg * upstream_grad).item() for lg in local_grads]
    #
    #     return np.reshape(np.asarray(final_grads), x.shape)
