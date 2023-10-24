"""
This script contains an abstract class for Parametrized layers (explaining param_layer file name)
"""

import numpy as np

from typing import List
from tinyBackProp.abstract_layer import Layer
from abc import ABC, abstractmethod


class ParamLayer(Layer, ABC):
    """
    This class is a child of the general 'Layer' class. Furthermore, it offers functionalities specific for
    Layers with parameters
    """

    def __init__(self):
        super().__init__()
        # each ParamLayer will have a weight parameter
        self.weight = None

    @abstractmethod
    def local_param_grads(self, x: np.ndarray) -> List[np.ndarray]:
        """
        This function calculates the derivative of the output with respect to every scalar in the weight tensor.
        Args:
            x: The input

        Returns: A list of numpy arrays of the same shape as the output, where each element in the list
        represents the derivative of the output with respect to the
        """
        pass

    def _reshape_grad(self, final_grads: list[float]) -> np.ndarray:
        """
        Given the derivative of the upstream gradient with respect ot each element
        Args:
            final_grads: This function receives a list L of scalars where the cardinality of L is the same
            as the cardinality of the parameters matrix
        Returns: a numpy array of the same shape as the weight Tensor.
        """
        grad_reshaped = np.reshape(np.asarray(final_grads), self.weight.shape)
        return grad_reshaped

    def param_grad(self, x: np.ndarray = None, upstream_grad: np.ndarray = None) -> np.ndarray:
        x = self.last_x if x is None else x

        if x is None:
            raise TypeError(f"the method is expecting non None input")

        self.last_x = x

        # first step generate the local gradient
        lgs = self.local_param_grads(x)
        # make sure the shape of the upstream_gram matches that of any given local grad
        if lgs[0].shape != upstream_grad.shape:
            raise ValueError((f"The upstream gradient is expected to be of the same shape as the local gradients\n "
                              f"Expected: {lgs[0].shape}. Found: {upstream_grad.shape}"))

        # the next step is to compute the upstream gradient with respect to each parameter
        final_grads = [np.sum(lg * upstream_grad).item() for lg in lgs]

        return self._reshape_grad(final_grads)

    def update(self, grad: np.ndarray, learning_rate: float) -> None:
        """
        This function updates the weight tensor using the Gradient descent update rule (inplace)
        Args:
            grad: The gradient of the output with respect to the parameters
            learning_rate: the step size

        Returns:
        """
        # make sure the grad is of the same shape as the weight matrix
        if grad.shape != self.weight.shape:
            raise ValueError((f"The same of the gradient and the weight matrix must be the same"
                              f"Found grad: {grad.shape} and weight: {self.weight.shape}"))

        self.weight = self.weight - grad * learning_rate

