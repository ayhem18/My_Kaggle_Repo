"""
This script contains the concrete implementation of the Linear Layer
"""

import random
import numpy as np

from .abstract_layer import Layer

# setting the seeds
np.random.seed(69)
random.seed(69)


class LinearLayer(Layer):
    def __init__(self, in_features: int, out_features: int, weight_matrix: np.ndarray):
        if weight_matrix is not None and weight_matrix.shape != (in_features, out_features):
            raise ValueError(f"if the weight matrix is explicitly passed. The dimensions must match")

        self.in_features = in_features
        self.out_features = out_features
        self.w = weight_matrix if weight_matrix is not None else np.random.rand(self.in_features, self.out_features)

    def _verify_input(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        if x.shape[-1] != self.in_features:
            raise ValueError(f"The vector is expected to be a row vector of length {self.n}\nFound:{x.shape}")

        return x

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self._verify_input(x)
        return x @ self.w

    def local_grad(self, x: np.ndarray):
        batch_size, x_m = x.shape

        def gradient_matrix(i, j):
            # the gradient matrix should have 0s in all columns different from j
            result = np.zeros(shape=(batch_size, self.out_features), dtype=np.float32)

            for k in range(batch_size):
                result[k, j] = x[k, i]

            # result[list(range(batch_size)), j] = x[list(range(batch_size)), i]
            # # the i-th column in the result should be the same as the j-th column of the 'x' matrix
            # result[list(range(batch_size)), i] = x[list(range(batch_size)), i]
            # set the i-th row of the result matrix to the j-th column of the 'x' matrix
            # for index in range(self.m):
            #     result[i, index] = x[j, index]
            return result

        grads = [gradient_matrix(i, j) for i in range(self.in_features) for j in range(self.out_features)]
        return grads

    def _reshape_grad(self, local_grads: list[float]) -> np.ndarray:
        grad_reshaped = np.reshape(np.asarray(local_grads), self.w.shape)
        return grad_reshaped
