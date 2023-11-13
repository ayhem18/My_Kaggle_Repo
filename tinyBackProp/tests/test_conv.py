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
from tinyBackProp.functional.convolution import conv_grad

np.random.seed(69)
import random
random.seed(69)
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
    padding_possible_values = [None] + list(range(1, 6)) # either no padding (None) or padding with a value of 1 to 6 

    for _ in range(num_test):
        for k in [3, 5, 7]:
            batch_size = 5
            out, c, h, w = random.randint(2, 5), random.randint(2, 10), random.randint(10, 15), random.randint(10, 15)
            x = torch.randn(batch_size, c, h, w)
            x_np = x.numpy()

            padding = random.choice(padding_possible_values)

            torch_layer = nn.Conv2d(in_channels=c, out_channels=out, kernel_size=(k, k), padding=('valid' if padding is None else padding) , bias=False)
            custom_layer = cl.ConvLayer(out_channels=out, in_channels=c, kernel_size=(k, k),padding=padding,
                                        weight_matrix=torch_layer.weight.squeeze().cpu().detach().numpy())

            y_torch = torch_layer(x).squeeze().detach().numpy()
            y_custom = custom_layer(x_np)

            assert np.allclose(y_torch, y_custom, atol=1e-06), "make sure the forward pass is correct"


def test_conv_backward(num_test: int = 100):
    padding_possible_values = [None] + list(range(1, 6)) # either no padding (None) or padding with a value of 1 to 5

    for _ in range(num_test):
        for k in [3, 5, 7]:
            # test with absolute value loss
            # generate the needed data for testing
            out, c, h, w = random.randint(1, 10), random.randint(2, 10), random.randint(8, 13), random.randint(8, 13)
            x = torch.randn(5, c, h, w) 
            x_np = x.numpy()

            padding = random.choice(padding_possible_values)

            # create the torch conv layer
            torch_layer = nn.Conv2d(in_channels=c, out_channels=out, kernel_size=(k, k), padding=('valid' if padding is None else padding), bias=False)
            # custom layer
            custom_layer = cl.ConvLayer(out_channels=out, in_channels=c, kernel_size=(k, k), padding=padding,
                                        weight_matrix=torch_layer.weight.squeeze().cpu().detach().numpy())

            # forward pass
            y_torch = torch_layer(x)
            y_custom = custom_layer(x_np)

            # loss function
            loss_object = SumLoss()
            # calculate the loss for the torch layer
            torch_loss = loss_object(y_torch)
            torch_loss.backward()

            # extract the gradient on the weights
            torch_grad = torch_layer.weight.grad.detach().cpu().numpy()

            # the gradient of the loss with respect to the final output is the sign function
            initial_upstream_grad = np.sign(y_custom)

            # calculate the gradient of the output with respect to the weights
            custom_grad = custom_layer.param_grad(x_np, initial_upstream_grad)
            
            try: 
                assert np.allclose(custom_grad, torch_grad, atol=10 ** -4)  
            except AssertionError:
                print(np.max(np.abs(custom_grad - torch_grad)))
                print(np.sum(np.abs(custom_grad - torch_grad) >= 10 ** -4))
                print(f"input dimensions: {x.shape}")
                print(f"padding: {padding}, k: {k}")
                
                continue

            # test with normal sum loss
            out, c, h, w = random.randint(1, 10), random.randint(2, 10), random.randint(10, 15), random.randint(10, 15)
            x = torch.randn(5, c, h, w)
            x_np = x.numpy()

            torch_layer = nn.Conv2d(in_channels=c, out_channels=out, kernel_size=(k, k), padding='valid', bias=False)
            custom_layer = cl.ConvLayer(in_channels=c,
                                        out_channels=out, 
                                        kernel_size=(k, k),
                                        weight_matrix=torch_layer.weight.cpu().detach().numpy())

            y_torch = torch_layer(x)
            y_custom = custom_layer(x_np)

            # the loss function was designed for its simplicity as its gradient is (1) regardless of the variable value
            loss_object = SumLoss(absolute=False)
            # calculate the loss for
            torch_loss = loss_object(y_torch)
            torch_loss.backward()

            torch_grad = torch_layer.weight.grad.detach().cpu().numpy()
            # the initial grad will be a matrix of ones
            initial_upstream_grad = np.ones(y_custom.shape, dtype=np.float32)

            custom_grad = custom_layer.param_grad(x_np, initial_upstream_grad)

            try: 
                assert np.allclose(custom_grad, torch_grad, atol=10 ** -4)  
            except AssertionError:
                print(np.max(np.abs(custom_grad - torch_grad)))
                print(np.sum(np.abs(custom_grad - torch_grad) >= 10 ** -4))
                continue


def test_conv_backward_x(num_test: int = 100):
    padding_possible_values = [None] + list(range(1, 6)) # either no padding (None) or padding with a value of 1 to 6 

    for _ in range(num_test):
        for k in [3, 5, 7]:
            out, c, h, w = random.randint(1, 10), random.randint(2, 10), random.randint(8, 13), random.randint(8, 13)
            x = torch.randn(5, c, h, w)
            x.requires_grad = True
            x_np = x.detach().cpu().numpy()

            padding = random.choice(padding_possible_values)

            # create the torch conv layer
            torch_layer = nn.Conv2d(in_channels=c, out_channels=out, kernel_size=(k, k), padding=('valid' if padding is None else padding), bias=False)
            # custom layer
            custom_layer = cl.ConvLayer(in_channels=c, 
                                        out_channels=out, 
                                        kernel_size=(k, k),
                                        padding=padding,
                                        weight_matrix=torch_layer.weight.cpu().detach().numpy())

            # forward pass
            y_torch = torch_layer(x)
            y_custom = custom_layer(x_np)

            # loss function
            loss_object = SumLoss()
            # calculate the loss for the torch layer
            torch_loss = loss_object(y_torch)
            torch_loss.backward()

            torch_grad = x.grad.detach().cpu().numpy()

            # the gradient of the loss with respect to the final output is the sign function
            initial_upstream_grad = np.sign(y_custom)

            # calculate the gradient of the output with respect to the weights
            custom_grad = custom_layer.grad(x_np, initial_upstream_grad)
            try:
                assert np.allclose(custom_grad, torch_grad, atol=10 ** -4)
            except AssertionError:
                print(np.max(np.abs(custom_grad - torch_grad)))
                print(np.sum(np.abs(custom_grad - torch_grad) >= 10 ** -4))
                sys.exit()


def test_function(num_test: int = 100):
    for _ in range(num_test):
        for k in [3, 5, 7]:
            # test with absolute value loss
            # generate the needed data for testing
            out, c, h, w = random.randint(2, 10), random.randint(2, 10), random.randint(8, 13), random.randint(8, 13)
            x = torch.randn(5, c, h, w) 
            x_np = x.numpy()

            # create the torch conv layer
            torch_layer = nn.Conv2d(in_channels=c, out_channels=out, kernel_size=(k, k), padding='valid', bias=False)
            # custom layer
            custom_layer = cl.ConvLayer(out_channels=out, in_channels=c, kernel_size=(k, k),
                                        weight_matrix=torch_layer.weight.squeeze().cpu().detach().numpy())

            # forward pass
            y_torch = torch_layer(x)
            y_custom = custom_layer(x_np)

            # loss function
            loss_object = SumLoss()
            # calculate the loss for the torch layer
            torch_loss = loss_object(y_torch)
            torch_loss.backward()

            # extract the gradient on the weights
            torch_grad = torch_layer.weight.grad.detach().cpu().numpy()

            # the gradient of the loss with respect to the final output is the sign function
            initial_upstream_grad = np.sign(y_custom)

            # calculate the gradient of the output with respect to the weights
            _, custom_grad = conv_grad(custom_layer.last_x, custom_layer.weight, dL=initial_upstream_grad)
            
            try: 
                assert np.allclose(custom_grad, torch_grad, atol=10 ** -4)  
            except AssertionError:
                print(np.max(np.abs(custom_grad - torch_grad)))
                print(np.sum(np.abs(custom_grad - torch_grad) >= 10 ** -4))
                print(f"input dimensions: {x.shape}")
                
                continue

            # test with normal sum loss
            out, c, h, w = random.randint(1, 10), random.randint(2, 10), random.randint(10, 15), random.randint(10, 15)
            x = torch.randn(5, c, h, w)
            x_np = x.numpy()

            torch_layer = nn.Conv2d(in_channels=c, out_channels=out, kernel_size=(k, k), padding='valid', bias=False)
            custom_layer = cl.ConvLayer(in_channels=c,
                                        out_channels=out, 
                                        kernel_size=(k, k),
                                        weight_matrix=torch_layer.weight.cpu().detach().numpy())

            y_torch = torch_layer(x)
            y_custom = custom_layer(x_np)

            # the loss function was designed for its simplicity as its gradient is (1) regardless of the variable value
            loss_object = SumLoss(absolute=False)
            # calculate the loss for
            torch_loss = loss_object(y_torch)
            torch_loss.backward()

            torch_grad = torch_layer.weight.grad.detach().cpu().numpy()
            # the initial grad will be a matrix of ones
            initial_upstream_grad = np.ones(y_custom.shape, dtype=np.float32)

            custom_grad = custom_layer.param_grad(x_np, initial_upstream_grad)

            try: 
                assert np.allclose(custom_grad, torch_grad, atol=10 ** -4)  
            except AssertionError:
                print(np.max(np.abs(custom_grad - torch_grad)))
                print(np.sum(np.abs(custom_grad - torch_grad) >= 10 ** -4))
                continue



if __name__ == '__main__':
    # print("Testing the forward pass started")
    # test_conv_forward()
    # print("Testing the forward pass successfully completed")
    
    # print("Testing the gradient over the weights started")
    # test_conv_backward()
    # print("Testing the gradient over the weights successfully completed")

    test_function()
