import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

home = os.path.dirname(os.path.realpath(__file__))
current = home

while 'tinyBackProp' not in os.listdir(current):
    current = Path(current).parent

sys.path.append(str(current))
sys.path.append(os.path.join(str(current)))

import tinyBackProp.conv_layer as cl

torch.manual_seed(69)
from torch import nn


class SumLoss(nn.Module):
    def __init__(self, absolute: bool = True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.abs = absolute

    def forward(self, x: torch.Tensor):
        if self.abs:
            return torch.sum(torch.abs(x))
        return torch.sum(x)


# the gradient of this function with respect to 'x' will always be a tensor of the same shape as 'x'
# where each derivative is the sign of the entry in 'x'


def test_conv_forward(num_test: int = 100):
    for _ in range(num_test):
        for k in [3, 5, 7]:
            batch_size = 5
            out, c, h, w = random.randint(2, 5), random.randint(2, 10), random.randint(10, 15), random.randint(10, 15)
            x = torch.randn(batch_size, c, h, w)
            x_np = x.numpy()

            torch_layer = nn.Conv2d(in_channels=c, out_channels=out, kernel_size=(k, k), padding='valid', bias=False)
            custom_layer = cl.ConvLayer(out_channels=out, in_channels=c, kernel_size=(k, k),
                                        weight_matrix=torch_layer.weight.squeeze().cpu().detach().numpy())

            y_torch = torch_layer(x).squeeze().detach().numpy()
            y_custom = custom_layer(x_np)

            assert np.allclose(y_torch, y_custom, atol=1e-06), "make sure the forward pass is correct"


def test_conv_backward(num_test: int = 100):
    for _ in range(num_test):
        for k in [3, 5, 7]:
            # test with absolute value loss
            # generate the needed data for testing
            out, c, h, w = random.randint(2, 5), random.randint(2, 10), random.randint(10, 15), random.randint(10, 15)
            x = torch.randn(c, h, w)
            x_np = x.numpy()

            # create the torch conv layer
            torch_layer = nn.Conv2d(in_channels=c, out_channels=out, kernel_size=(k, k), padding='valid', bias=False)
            # custom layer
            custom_layer = cl.ConvLayer(out_channels=out, in_channels=c, kernel_size=(k, k),
                                        weight_matrix=torch_layer.weight.squeeze().cpu().detach().numpy())

            # forward pass
            y_torch = torch_layer(x)
            y_custom = custom_layer(x_np).squeeze()

            # loss function
            loss_object = SumLoss()
            # calculate the loss for the torch layer
            torch_loss = loss_object(y_torch)
            torch_loss.backward()

            # extract the gradient on the weights
            torch_grad = torch_layer.weight.grad.detach().squeeze().cpu().numpy()

            # the gradient of the loss with respect to the final output is the sign function
            initial_upstream_grad = np.sign(y_custom)

            # calculate the gradient of the output with respect to the weights
            custom_grad = custom_layer.param_grad(x_np, initial_upstream_grad)

            assert np.allclose(custom_grad, torch_grad, atol=10 ** -5)

            # test with normal sum loss
            c, h, w = random.randint(2, 10), random.randint(10, 15), random.randint(10, 15)
            x = torch.randn(c, h, w)
            x_np = x.numpy()

            torch_layer = nn.Conv2d(in_channels=c, out_channels=1, kernel_size=(k, k), padding='valid', bias=False)
            custom_layer = cl.ConvLayer(in_channels=c, kernel_size=(k, k),
                                        weight_matrix=torch_layer.weight.squeeze().cpu().detach().numpy())

            y_torch = torch_layer(x)
            y_custom = custom_layer(x_np).squeeze()

            # the loss function was designed for its simplicity as its gradient is (1) regardless of the variable value
            loss_object = SumLoss(absolute=False)
            # calculate the loss for
            torch_loss = loss_object(y_torch)
            torch_loss.backward()

            torch_grad = torch_layer.weight.grad.detach().squeeze().cpu().numpy()
            # the initial grad will be a matrix of ones
            initial_upstream_grad = np.ones(y_custom.shape, dtype=np.float32)

            custom_grad = custom_layer.param_grad(x_np, initial_upstream_grad)
            assert np.allclose(custom_grad, torch_grad, atol=10 ** -5)




def test_conv_backward_x(num_test: int = 100):
    for _ in range(num_test):
        for k in [3, 5, 7]:
            c, h, w = random.randint(2, 10), random.randint(10, 15), random.randint(10, 15)
            x = torch.randn(c, h, w)
            x.requires_grad = True
            x_np = x.detach().cpu().numpy()

            # create the torch conv layer
            torch_layer = nn.Conv2d(in_channels=c, out_channels=1, kernel_size=(k, k), padding='valid', bias=False)
            # custom layer
            custom_layer = cl.ConvLayer(in_channels=c, kernel_size=(k, k),
                                        weight_matrix=torch_layer.weight.squeeze().cpu().detach().numpy())

            # forward pass
            y_torch = torch_layer(x)
            y_custom = custom_layer(x_np).squeeze()

            # loss function
            loss_object = SumLoss()
            # calculate the loss for the torch layer
            torch_loss = loss_object(y_torch)
            torch_loss.backward()

            torch_grad = x.grad

            # the gradient of the loss with respect to the final output is the sign function
            initial_upstream_grad = np.sign(y_custom)

            # calculate the gradient of the output with respect to the weights
            custom_grad = custom_layer.grad(x_np, initial_upstream_grad)

            assert np.allclose(custom_grad, torch_grad, atol=10 ** -5)


if __name__ == '__main__':
    test_conv_forward()
    # test_conv_backward()
    # test_conv_backward_x()
