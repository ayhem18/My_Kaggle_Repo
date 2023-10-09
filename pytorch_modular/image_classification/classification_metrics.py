"""
This script contains several functionalities related to model evaluation and mainly metrics
such as 'accuracy', 'f1', 'precision', 'recall'
"""

import torch

ACCURACY = 'accuracy'


def accuracy(y_pred: torch.tensor, y: torch.tensor) -> float:
    # squeeze values if needed
    value = (y_pred == y).type(torch.float32).mean().item()
    return value
