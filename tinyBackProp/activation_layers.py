"""
This script contains the concrete implementations of a number of layers used in this small Deep Learning library.
"""
from _ast import List

import numpy as np
from .abstract_layer import Layer


# noinspection PyUnresolvedReferences
class SoftmaxLayer(Layer):

    def __init__(self, normalization: bool = True) -> None:
        super().__init__()
        self.norm = normalization

    def _verify_input(self, x: np.ndarray) -> np.ndarray:
        if x.ndim > 2:
            raise ValueError(f"The input is expected to be at most 2 dimensional.\nFound: {len(x.shape)} dimensions")
        return x

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.squeeze(self._verify_input(x))
        # the main idea behind 'normalization' is to avoid numerical overflow with the softmax function
        # (mainly with the denominator as a sum of exponential functions). The output of softmax for (x1, x2, ... xn) is the same as the output for
        # (x1 - C, x2 - C, ... xn - C) where C is any constant. (other operations such as division will alter the output)
        # consider the following link for the mathematical details : https://jaykmody.com/blog/stable-softmax/

        norm_factor = max(x) if self.norm else 0
        # if normalization is False, then 'x' is the same as the original output,
        # otherwise the maximum element will be subtracted from each element
        x = x - norm_factor
        sum_exp = np.sum(np.exp(x))
        result = np.exp(x) / sum_exp
        # make sure the final output is 2-dimensional
        return result if len(result.shape) == 2 else np.expand_dims(result, axis=-1)

    def local_grad(self, x: np.ndarray) -> List[np.ndarray]:
        # forward pass
        s = self.forward(x)
        # let's build the jacobian matrix
        jacobian = - s @ np.transpose(s)

        for i in range(len(x)):
            jacobian[i][i] += s[i][0]

        # each the entry (i, j) will represent the derivative of S_i with respect to x_j
        # the next step is simply convert the jacobian matrix to a list of numpy array

        return [jacobian[list(range(len(s))), i] for i in range(len(s))]

    def _reshape_grad(self, local_grads: List[float]) -> np.ndarray:
        # simply convert the list to a column numpy array vector
        result = np.expand_dims(np.array(local_grads), axis=-1)
        if result.shape != (len(local_grads), 1):
            raise ValueError(f"Make sure the result is a column vector.\n"
                             f"Expected: {(len(local_grads), 1)}. Found: {result.shape}")
        return result


class ReLULayer(Layer):
    def __init__(self) -> None:
        super().__init__()

    def _verify_input(self, x: np.ndarray) -> np.ndarray:
        if x.ndim > 2:
            raise ValueError(f"The input is expected to be at most 2 dimensional.\nFound: {len(x.shape)} dimensions")
        return np.expand_dims(x, axis=-1) if x.ndim == 1 else x

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self._verify_input(x)
        return x * (x > 0)

    def local_grad(self, x: np.ndarray) -> list[np.ndarray]:
        x = self._verify_input(x)
        jacobian = np.eye(N=len(x))
        for i, v in enumerate(x):
            jacobian[i][i] = float(v > 0)

        return [jacobian[list(range(len(x))), i] for i in range(len(x))]

    def _reshape_grad(self, local_grads: list[float]) -> np.ndarray:
        # simply convert the list to a column numpy array vector
        result = np.expand_dims(np.array(local_grads), axis=-1)
        if result.shape != (len(local_grads), 1):
            raise ValueError(f"Make sure the result is a column vector.\n"
                             f"Expected: {(len(local_grads), 1)}. Found: {result.shape}")
        return result
