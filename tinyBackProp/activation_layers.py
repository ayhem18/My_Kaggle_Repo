"""
This script contains the concrete implementations of a number of layers used in this small Deep Learning library.
"""

from typing import List, Union
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

    def forward(self, x: np.ndarray = None) -> np.ndarray:
        super().forward(x)
        # set the default 'x' if needed
        x = self.last_x if x is None else x

        x = self._verify_input(x)
        # the main idea behind 'normalization' is to avoid numerical overflow with the softmax function
        # (mainly with the denominator as a sum of exponential functions). The output of softmax for (x1, x2, ... xn) is the same as the output for
        # (x1 - C, x2 - C, ... xn - C) where C is any constant. (other operations such as division will alter the output)
        # consider the following link for the mathematical details : https://jaykmody.com/blog/stable-softmax/

        norm_factor = np.max(x, axis=1, keepdims=True) if self.norm else 0
        # if normalization is False, then 'x' is the same as the original output,
        # otherwise the maximum element will be subtracted from each element
        x = x - norm_factor
        sum_exp = np.sum(np.exp(x), axis=1, keepdims=True)
        result = np.exp(x) / sum_exp
        return result

    def local_x_grad(self, x: np.ndarray) -> List[np.ndarray]:
        """This function will return the jacobian matrix

        Args:
            x (np.ndarray): _description_

        Returns:
            List[np.ndarray]: _description_
        """
        if x.ndim != 2 or x.shape[0] != 1:
            raise ValueError(f"The input is expected to be 2 dimensional")

        # forward pass  
        s = self.forward(x)
        # let's build the jacobian matrix
        jacobian = - np.transpose(s) @ s

        for i in range(s.shape[1]):
            jacobian[i][i] += s[0][i]

        # each the entry (i, j) will represent the derivative of S_i with respect to x_j
        # the next step is simply convert the jacobian matrix to a list of numpy array

        return jacobian

        # return [jacobian[list(range(len(s))), i] for i in range(len(s))]

    def _reshape_grad(self, local_grads: List[float]) -> np.ndarray:
        # simply convert the list to a column numpy array vector
        result = np.expand_dims(np.array(local_grads), axis=-1)
        if result.shape != (len(local_grads), 1):
            raise ValueError(f"Make sure the result is a column vector.\n"
                             f"Expected: {(len(local_grads), 1)}. Found: {result.shape}")
        return result

    def grad(self, x: np.ndarray = None, upstream_grad: np.ndarray = None) -> np.ndarray:
        x = self.last_x if x is None else x
        if x is None:
            raise TypeError(f"the method is expecting non None input")
        self.last_x = x

        # the main idea here is that upstream_grad must be of the same shape as 'x'
        if upstream_grad.shape != x.shape:
            raise ValueError(f"The upstream gradient is expected to be of the same shape as 'x'")

        result = [(sample_up_grad @ self.local_x_grad(np.expand_dims(sample, axis=0))).tolist() for sample, sample_up_grad
                  in zip(x, upstream_grad)]

        return np.asarray(result)

class ReLULayer(Layer):
    def __init__(self) -> None:
        super().__init__()

    def _verify_input(self, x: np.ndarray) -> np.ndarray:
        return np.expand_dims(x, axis=-1) if x.ndim == 1 else x

    def forward(self, x: np.ndarray = None) -> np.ndarray:
        super().forward(x)
        x = self.last_x if x is None else x
        x = self._verify_input(x)
        return x * (x > 0)

    def local_x_grad(self, x: np.ndarray) -> Union[list[np.ndarray], np.ndarray]:
        # the activation layers are special layers in the sense that they do not have explicit parameters
        # the input represents the layer's parameters
        return np.asarray((x > 0)).astype(np.float32)

    def _reshape_grad(self, local_grads: list[float]) -> np.ndarray:
        # simply convert the list to a column numpy array vector
        result = np.expand_dims(np.array(local_grads), axis=-1)
        if result.shape != (len(local_grads), 1):
            raise ValueError(f"Make sure the result is a column vector.\n"
                             f"Expected: {(len(local_grads), 1)}. Found: {result.shape}")
        return result

    def grad(self, x: np.ndarray = None, upstream_grad: np.ndarray = None) -> np.ndarray:
        # set the default 'x' if needed
        x = self.last_x if x is None else x

        # the main idea here is that upstream_grad must be of the same shape as 'x'
        if upstream_grad.shape != x.shape:
            raise ValueError(f"The upstream gradient is expected to be of the same shape as 'x'")

        return self.local_x_grad(x) * upstream_grad

class SigmoidLayer(Layer):
    def __init__(self) -> None:
        super().__init__()


    def _verify_input(self, x: np.ndarray) -> np.ndarray:
        return np.expand_dims(x, axis=-1) if x.ndim == 1 else x


    def forward(self, x: np.ndarray = None) -> np.ndarray:
        super().forward(x)
        x = self._verify_input(x)
        return (1 / (1 + np.exp(-x)))


    def grad(self, x: np.ndarray = None, upstream_grad: np.ndarray = None) -> np.ndarray:
        x = self.last_x if x is None else x

        # the main idea here is that upstream_grad must be of the same shape as 'x'
        if upstream_grad.shape != x.shape:
            raise ValueError(f"The upstream gradient is expected to be of the same shape as 'x'")

        return self.forward(x) * (1 - self.forward(x)) * upstream_grad

