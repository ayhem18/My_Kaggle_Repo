import random
import torch

import numpy as np

from typing import Union
from .abstract_layer import Loss

random.seed(69)
torch.manual_seed(69)
np.random.seed(69)

num = Union[float, int]


class CrossEntropyLoss(Loss):
    def __init__(self,
                 num_classes: int,
                 reduction: str = 'mean',
                 epsilon: float = 10 ** - 8):
        if num_classes <= 2:
            raise ValueError(f"The cross entropy loss expects more than 2 classes.\nFor binary classification problems"
                             f" Please use BinaryCrossEntropyLoss")
        self.n_classes = num_classes
        self.reduction = reduction
        self.epsilon = epsilon

    def _verify_input(self, y_pred: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # make sure the y_pred shape is at (None, num_classes)
        if len(y_pred.shape) != 2 or y_pred.shape[-1] <= 2:
            raise ValueError(f'The predictions are expected to be of shape {(None, "n")} where {"n"} is larger than 2')

        # make sure the predictions are probabilities
        if not (np.all(y_pred >= 0) and np.isclose(np.ones(shape=(len(y_pred), 1)), np.sum(y_pred, axis=0))):
            raise ValueError(f"the predictions are expected to represent probabilities.")

        # make sure there are many predictions as labels
        if len(y_pred) != len(y):
            raise ValueError(f"The loss is expecting as many examples as labels")

        # make sure all the target are less than 'num_classes':
        if not np.all((self.n_classes > y) & (y >= 0)):
            raise ValueError((f"The targets are expected to be in the range [0, number of classes]\n"
                              f"Found: {y[(y < 0) | (y >= self.n_classes)][0]}"))

        # now we are ready to proceed
        return y_pred, y

    def forward(self, y_pred: np.ndarray, y: np.ndarray, reduction: str = 'mean') -> Union[float, np.ndarray]:
        y_pred += self.epsilon * (y_pred <= self.epsilon)
        result = -np.log(y_pred[list(range(len(y_pred))), y])
        return np.mean(result) if reduction == 'mean' else result

    def grad(self, y_pred: np.ndarray, y_true: np.ndarray, reduction: str = 'mean') -> np.ndarray:
        # the idea here is to create a one-hot encoding of the labels
        y_true_ohe = np.zeros(y_pred.shape, dtype=int)
        y_true_ohe[np.arange(y_true.size), y_true] = 1

        # add self.epsilon to zero entries
        y_pred += self.epsilon * (y_pred <= self.epsilon)

        grad_total = (-1 / y_pred) * y_true_ohe

        if reduction == 'mean':
            grad_total /= len(y_true)
        return grad_total


class MSELoss(Loss):
    def __init__(self,
                 reduction: str = 'mean'):
        self.reduction = reduction

    def _verify_input(self, y_pred: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # this function just makes sure 
        # 1. y_pred is 2 dimensional
        # 2. y and y_pred are of the same shape
        
        if y_pred.ndim > 2: 
            raise ValueError(f"The function expects an input with at most 2 dimensions. Found: {y_pred.shape}")
        
        y_pred = np.expand_dims(y_pred, axis=-1) if y_pred.ndim == 1 else y_pred
        y = np.expand_dims(y, axis=-1) if y.ndim == 1 else y

        if y_pred.shape != y.shape:
            raise ValueError(f"Make sure the pre")  
        
        return y_pred, y

    def forward(self, y_pred: np.ndarray, y: np.ndarray) -> Union[float, np.ndarray]:
        y_pred, y = self._verify_input(y, y_pred)
        loss = (y_pred - y) ** 2
        if self.reduction == 'mean':
            return np.mean(loss).item()
        else: 
            return np.sum(loss).item()
        

    def grad(self, y_pred: np.ndarray, y: np.ndarray, ) -> np.ndarray:
        y_pred, y = self._verify_input(y, y_pred)
        # MSE is a simple quadratic function
        g = 2 * (y - y_pred)

        if self.reduction == 'mean':
            # averaging by the number of entries in 'y'
            return g / len(y.reshape(-1))
        
        return g

