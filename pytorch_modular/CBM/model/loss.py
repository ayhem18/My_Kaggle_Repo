"""
This script contains the main definition of the CBM loss
"""

import torch

from torch import nn
from typing import Union, Tuple


class CBMLoss(nn):
    def __init__(self,
                 alpha: float,
                 concepts_loss: nn.Module = None):
        # alpha is the balancing parameter
        self.alpha = alpha
        # the default loss is Cross Entropy, the pytorch implementation supports soft labels
        self.concepts_loss = nn.CrossEntropyLoss() if concepts_loss is None else concepts_loss

        # make sure the loss is a nn.Module
        if not isinstance(concepts_loss, nn.Module):
            raise TypeError(f"The loss must be of type {nn.Module}. Found: {type(concepts_loss)}")

    def forward(self,
                concept_preds: torch.Tensor,
                concepts_true: torch.Tensor,
                y_pred: torch.Tensor,
                y_true: torch.Tensor,
                return_all: bool = False) -> Union[nn.Module, Tuple[nn.Module, nn.Module, nn.Module]]:

        # for concepts predictions, the following criterion should be satisfied
        if concept_preds.shape != concepts_true.shape:
            raise ValueError((f"The concepts labels and concepts logits are expected to be of matching shapes\n"
                              f"Found logits: {concept_preds.shape}, labels: {concepts_true.shape}"))

        if torch.sum((concepts_true < 0)) > 0:
            raise ValueError(f"Make sure each value in all concept labels are non-negative")

        # make sure the concept labels can be seen / considered as probability distributions
        if not torch.allclose(torch.sum(concept_preds, dim=1), torch.ones(), atol=10 ** -3):
            raise ValueError(f"The entries in concept labels do not sum up to 1")

            # now we are ready to proceed
        class_loss = nn.CrossEntropyLoss()(y_pred, y_true)
        concept_loss = self.concepts_loss(concept_preds, concepts_true)
        final_loss = class_loss + self.alpha * concept_loss

        if return_all:
            return class_loss, concept_loss, final_loss

        return final_loss
