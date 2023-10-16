"""
This script contains the definition of the main class in this very small Deep Learning library
"""

from typing import List, Union, Tuple
import numpy as np

from abc import ABC, abstractmethod


class Layer(ABC):

    @abstractmethod
    def _verify_input(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        # given an input, this functions simply returns the output of the layer
        # nothing too complicated
        pass

    @abstractmethod
    def local_grad(self, x: np.ndarray) -> list[np.ndarray]:
        """This function generates the local gradients of a given layer with respect to the passed input.
        Assuming the layer has N parameters, w_1, w_2,..., w_N, then the local gradient will return a List of N elements
        where each element is an output-like np.array representing the gradient of the parameter w_i with respect to each element in the output.

        This approach generalizes the concept of local gradient to any input / output dimensions.

        Args:
            x (np.ndarray): the given input

        Returns:
            List[np.ndarray]: 
        """
        pass

    @abstractmethod
    def _reshape_grad(self, local_grads: list[float]) -> np.ndarray:
        """Given the gradients of the final loss with respect to each parameter, this function reshapes the result to match
        the layer's parameter.

        Args:
            local_grads (List[float]): _description_

        Returns:
            np.ndarray: _description_
        """
        pass

    def grad(self, x: np.ndarray, upstream_grad: np.ndarray = None) -> np.ndarray:
        """This function generates the gradient of the final loss (supposedly the layer belongs to a computational graph with a loss at the root).
        The output will be a numpy array with the same shape as the input where each entry represents the derivative of the loss with respect to the 
        parameter in question.

        Args:
            x (np.ndarray): the input
            upstream_grad (np.ndarray, optional): the upstream gradient: Must be of the same shape as the layer's output. Defaults to None.

        Returns:
            np.ndarray: the derivative of the final loss with respect to the layer's parameters
        """
        # first step generate the local gradient
        local_grads = self.local_grad(x)
        # make sure the shape of the upstream_gram matches that of any given local grad
        if local_grads[0].shape != upstream_grad.shape:
            raise ValueError((f"The upstream gradient is expected to be of the same shape as the local gradients\n "
                              f"Expected: {local_grads[0].shape}. Found: {upstream_grad.shape}"))

        # the next step is to compute the upstream gradient with respect to each parameter
        final_grads = [np.sum(lg * upstream_grad).item() for lg in local_grads]

        return self._reshape_grad(final_grads)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class Loss(ABC):
    @abstractmethod
    def _verify_input(self, y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> Union[float, np.ndarray]:
        pass

    @abstractmethod
    def grad(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        This function calculates the derivative of the loss with respect to each prediction
        The output is of the same shape as the 'y_pred' argument
        Args:
            y_pred:
            y_true:

        Returns:

        """
        pass

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> Union[float, np.ndarray]:
        return self.forward(y_pred, y_true)
