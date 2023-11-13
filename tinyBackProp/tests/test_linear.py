import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

home = os.path.dirname(os.path.realpath(__file__))
current = home
while 'tinyBackProp' not in os.listdir(current):
    current = Path(current).parent

sys.path.append(str(current))
sys.path.append(os.path.join(str(current)))

from tinyBackProp.linear_layer import LinearLayer

torch.manual_seed(69)


class SumLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        return torch.sum(torch.abs(x))


# the gradient of this function with respect to 'x' will always be a tensor of the same shape as 'x'
# where each derivative is the sign of the entry in 'x'


def test_linear_forward(num_test: int = 100):
    for _ in range(num_test):
        num_classes = random.randint(3, 20)
        num_samples = random.randint(10, 100)
        x = torch.randn(num_samples, num_classes) * 2
        x_np = x.numpy()

        out_feats = random.randint(5, 20) 

        torch_linear = nn.Linear(in_features=num_classes, out_features=out_feats, bias=False)
        w = torch_linear.weight.detach().numpy()
        custom_linear = LinearLayer(in_features=num_classes, out_features=out_feats, weight_matrix=w)

        y_torch = torch_linear(x).detach().numpy()
        y_custom = custom_linear(x_np)

        assert np.allclose(y_torch, y_custom, atol=1e-06), "make sure the forward pass is correct"


def test_linear_backward(num_test: int = 100):
    for _ in range(num_test):
        num_classes = random.randint(3, 20)
        num_samples = random.randint(10, 100)
        x = torch.randn(num_samples, num_classes) * 2
        x_np = x.numpy()

        torch_linear = nn.Linear(in_features=num_classes, out_features=num_samples, bias=False)
        w = torch_linear.weight.detach().numpy()
        custom_linear = LinearLayer(in_features=num_classes, out_features=num_samples, weight_matrix=w)

        y_torch = torch_linear(x)
        y_custom = custom_linear(x_np)

        loss_object = SumLoss()
        # calculate the loss for
        torch_loss = loss_object(y_torch)
        torch_loss.backward()

        torch_grad = torch_linear.weight.grad

        # the initial grad will be a matrix of ones
        initial_upstream_grad = np.sign(y_custom)

        custom_grad = custom_linear.param_grad(x_np, initial_upstream_grad)

        assert np.allclose(custom_grad, torch_grad, atol=10**-5)


def test_linear_backward_x(num_test: int = 100):

    for _ in range(num_test):
        num_classes = random.randint(3, 20)
        num_samples = random.randint(10, 100)
        x = torch.randn(num_samples, num_classes) * 2
        x_np = x.numpy()
        x.requires_grad = True

        torch_linear = nn.Linear(in_features=num_classes, out_features=num_samples, bias=False)
        w = torch_linear.weight.detach().numpy()
        custom_linear = LinearLayer(in_features=num_classes, out_features=num_samples, weight_matrix=w)

        # forward pass
        y_torch = torch_linear(x)
        y_custom = custom_linear(x_np)

        # loss function
        loss_object = SumLoss()
        # calculate the loss for the torch layer
        torch_loss = loss_object(y_torch)
        torch_loss.backward()

        torch_grad = x.grad.detach().cpu().numpy()

        # the gradient of the loss with respect to the final output is the sign function
        initial_upstream_grad = np.sign(y_custom)

        # calculate the gradient of the output with respect to the weights
        custom_grad = custom_linear.grad(x_np, initial_upstream_grad)
        
        try:
            assert np.allclose(custom_grad, torch_grad, atol=10 ** -4)
        except AssertionError:
            print(np.max(np.abs(custom_grad - torch_grad)))
            print(np.sum(np.abs(custom_grad - torch_grad) >= 10 ** -4))
            sys.exit()



if __name__ == '__main__':
    test_linear_forward()
    test_linear_backward()
    test_linear_backward_x()