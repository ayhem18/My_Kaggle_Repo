"""
This script contains the definition of the main class in this very small Deep Learning library
"""

from typing import List, Union, Tuple
import numpy as np

from abc import ABC, abstractmethod


class Layer(ABC):

    def __init__(self) -> None:
        super().__init__()
        self.last_x = None

    @abstractmethod
    def _verify_input(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def forward(self, x: np.ndarray = None):
        # given an input, this functions simply returns the output of the layer
        # nothing too complicated
        x = self.last_x if x is None else x
        if x is None:
            raise TypeError(f"the method is expecting non None input")
        self.last_x = x

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

    @abstractmethod
    def grad(self, x: np.ndarray = None, upstream_grad: np.ndarray = None) -> np.ndarray:
        pass

    @abstractmethod
    def update(self, grad: np.ndarray, learning_rate: float):
        pass

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
