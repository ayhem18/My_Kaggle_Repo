"""
This script contains the definition of the main class in this very small Deep Learning library
"""

from typing import Union, Tuple
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

    def local_x_grad(self, x: np.ndarray):
        """
        This function generates the gradient of the output with respect to the input.
        Args:
            x: the input
        Returns: a List of numpy arrays. Each with the same shape as the input representing the gradient
        of the output with respect to an entry in 'x'
        """
        pass

    def grad(self, x: np.ndarray = None, upstream_grad: np.ndarray = None) -> np.ndarray:
        """
        This function generates the gradient of the output with respect to the input:

        Assuming we have Z = L(X) where Z is the output and L is the forward function
        then this function returns a tensor with the same shape as the input where each entry
        computes the derivative of each input entry on the final upstream gradient
        """

        x = self.last_x if x is None else x

        if x is None:
            raise TypeError(f"the method is expecting non-None input")

        self.last_x = x

        # generate the gradient of the output with respect to x
        local_grads = self.local_x_grad(x)
        # make sure the shape of the upstream_gram matches that of any given local grad
        if local_grads[0].shape != upstream_grad.shape:
            raise ValueError((f"The upstream gradient is expected to be of the same shape as the local gradients\n "
                              f"Expected: {local_grads[0].shape}. Found: {upstream_grad.shape}"))

        # local_grads contains the gradient of the output with respect to the output: goi
        # final_grads should contain the upstream gradient with respect to the input: ugi
        # : the sum of the element-wise multiplication between the goi and ugi

        final_grads = [np.sum(lg * upstream_grad).item() for lg in local_grads]

        # reshape the final grads to match the input.
        return np.reshape(np.asarray(final_grads), x.shape)

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
