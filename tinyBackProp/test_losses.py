"""
"""

import os
import sys

import torch
import random

import numpy as np

from pathlib import Path
from torch import nn

home = os.path.dirname(os.path.realpath(__file__))
current = home
while 'tinyBackProp' not in os.listdir(current):
    current = Path(current).parent

sys.path.append(str(current))
sys.path.append(os.path.join(str(current)))

from tinyBackProp.losses import CrossEntropyLoss


def test_CEL_forward(num_test: int = 100):
    for _ in range(num_test):
        loss = nn.NLLLoss(reduction='none')
        num_classes = random.randint(3, 20)
        num_samples = random.randint(10, 100)

        x = torch.rand(num_samples, num_classes)
        # make sure to normalize so values are in [0, 1]
        x = x / torch.sum(x, dim=0)
        x_np = x.numpy()

        y_np = np.array([random.randint(0, num_classes - 1) for _ in range(num_samples)])
        y = torch.from_numpy(y_np)

        # torch does not provide functions that would calculate the cross entropy loss directly from predictions.
        # nn.CrossEntropy loss expects logits (the numbers before applying softmax)
        # nn.NLLLoss expects the logarithm of predictions which explains torch.log(x) in the loss call below

        torch_loss = loss(torch.log(x), y)
        torch_loss = torch_loss.numpy()
        loss_obj = CrossEntropyLoss(num_classes=num_classes)
        custom_loss = loss_obj.forward(x_np, y_np, reduction='none')
        # convert torch loss to numpy
        try:
            tl = torch_loss.item()
            assert abs(tl - custom_loss) <= 10 ** -5, "The loss for the scalar case is not correct"
        except:
            # this block of code is reached for the vector case
            assert np.allclose(torch_loss, custom_loss), "The loss for the vectorized case is not correct"


def test_CEL_backward(num_test: int = 100):
    for _ in range(num_test):
        num_classes = random.randint(3, 20)
        num_samples = random.randint(10, 100)

        x = torch.rand(num_samples, num_classes)
        # make sure to normalize so values are in [0, 1]
        x = x / torch.sum(x, dim=0)
        # set the grad to True
        x.requires_grad = True
        # now 'x' can be interpreted as a probability distribution
        x_np = x.detach().numpy()

        y_np = np.array([random.randint(0, num_classes - 1) for _ in range(num_samples)])
        y = torch.from_numpy(y_np)

        torch_cle = nn.NLLLoss()
        torch_loss = torch_cle(torch.log(x), y)
        torch_loss.backward()

        torch_grad = x.grad.numpy()

        custom_loss_obj = CrossEntropyLoss(num_classes=num_classes)
        custom_grad = custom_loss_obj.grad(y_pred=x_np, y_true=y_np)

        assert np.allclose(custom_grad, torch_grad), "Please make sure the gradient are computed correctly"


if __name__ == '__main__':
    test_CEL_forward()
    test_CEL_backward()
    # we are confident that both forward and backward calls are correct
