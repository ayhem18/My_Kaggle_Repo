"""
This script contains functionalities to test the activation layers
"""

import os, sys

import numpy as np
import torch
import random

from torch import nn
from pathlib import Path

home = os.path.dirname(os.path.realpath(__file__))
current = home
while 'tinyBackProp' not in os.listdir(current):
    current = Path(current).parent

sys.path.append(str(current))
sys.path.append(os.path.join(str(current)))

from tinyBackProp.activation_layers import ReLULayer, SoftmaxLayer, SigmoidLayer
from tinyBackProp.flatten import FlattenLayer

random.seed(69)
torch.manual_seed(69)


# the gradient of this function with respect to 'x' will always be a tensor of ones
class SumLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        return torch.sum(x)


# the gradient of this function with respect to 'x' will always be a tensor of the same shape as 'x'
# where each derivative is the sign of the entry in 'x'
class AbsSumLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        return torch.sum(torch.abs(x))


def test_relu_forward(num_test: int = 100):
    for _ in range(num_test):
        num_classes = random.randint(3, 20)
        # num_samples = random.randint(10, 100)
        x = torch.randn(1, num_classes) * 2
        x = x - torch.mean(x, dim=0)
        x_np = x.numpy()

        relu_torch = nn.ReLU()
        y_torch = relu_torch(x)

        custom_relu = ReLULayer()
        custom_y = custom_relu(x_np)

        assert np.allclose(custom_y, y_torch)


def test_relu_backward(num_test: int = 1000):
    for _ in range(num_test):
        num_classes = random.randint(3, 20)
        num_samples = random.randint(10, 100)
        x = torch.randn(num_samples, num_classes) * 2
        x = x - torch.mean(x, dim=0)

        x.requires_grad = True
        x_np = x.detach().numpy()

        custom_relu = ReLULayer()
        custom_y = custom_relu(x_np)

        loss_obj = SumLoss()
        loss = loss_obj(nn.ReLU()(x))
        loss.backward()
        torch_grad = x.grad

        initial_upstream_grad = np.ones(shape=custom_y.shape) # np.sign(custom_y)
        custom_grad = custom_relu.grad(x_np, initial_upstream_grad)

        assert np.allclose(custom_grad, torch_grad), "Please make sure the grad of Relu works correctly"

        x = torch.randn(num_samples, num_classes) * 2
        x = x - torch.mean(x, dim=0)

        x.requires_grad = True
        x_np = x.detach().numpy()

        custom_relu = ReLULayer()
        custom_y = custom_relu(x_np)

        loss_obj = AbsSumLoss()
        loss = loss_obj(nn.ReLU()(x))
        loss.backward()
        torch_grad = x.grad

        initial_upstream_grad = np.sign(custom_y)
        custom_grad = custom_relu.grad(x_np, initial_upstream_grad)

        assert np.allclose(custom_grad, torch_grad), "Please make sure the grad of Relu works correctly"


def test_soft_forward(num_test: int = 1000):
        for _ in range(num_test):
            num_classes = random.randint(3, 20)
            num_samples = random.randint(10, 100)
            x = torch.randn(num_samples, num_classes)
            x = x - torch.mean(x, dim=0)
            x_np = x.numpy()

            relu_torch = nn.Softmax(dim=1)
            y_torch = relu_torch(x)

            custom_soft = SoftmaxLayer()
            custom_y = custom_soft(x_np)

            assert np.allclose(custom_y, y_torch)


def test_soft_backward(num_test: int = 100):
        
        for _ in range(num_test):
            num_classes = random.randint(3, 20)
            num_samples = random.randint(10, 100)
            x = torch.randn(1, num_classes) * 2
            x = x - torch.mean(x, dim=0)

            x.requires_grad = True
            x_np = x.detach().numpy()

            custom_relu = SoftmaxLayer()
            custom_y = custom_relu(x_np)

            loss_obj = SumLoss()
            loss = loss_obj(nn.Softmax(dim=1)(x))
            loss.backward()
            torch_grad = x.grad

            initial_upstream_grad = np.ones(shape=custom_y.shape) # np.sign(custom_y)
            custom_grad = custom_relu.grad(x_np, initial_upstream_grad)

            assert np.allclose(custom_grad, torch_grad, atol=10**-6), "Please make sure the grad of Relu works correctly"

            x = torch.randn(num_samples, num_classes) * 2
            x = x - torch.mean(x, dim=0)

            x.requires_grad = True
            x_np = x.detach().numpy()

            custom_relu = SoftmaxLayer()
            custom_y = custom_relu(x_np)

            loss_obj = SumLoss()
            loss = loss_obj(nn.Softmax(dim=1)(x))
            loss.backward()
            torch_grad = x.grad

            initial_upstream_grad = np.sign(custom_y)
            custom_grad = custom_relu.grad(x_np, initial_upstream_grad)

            assert np.allclose(custom_grad, torch_grad, atol=10**-6), "Please make sure the grad of Relu works correctly"


def test_sigmoid_forward(num_test: int = 100):
    for _ in range(num_test):
        num_samples = random.randint(3, 50)
        dim = random.randint(5, 20)        

        x = torch.randn(num_samples, dim) * 2
        x = x - torch.mean(x, dim=0)
        x_np = x.numpy()

        sigmoid_torch = nn.Sigmoid()
        y_torch = sigmoid_torch(x)

        custom_relu = SigmoidLayer()
        custom_y = custom_relu(x_np)

        assert np.allclose(custom_y, y_torch), "Make sure sigmoied is implemented correctly"


def test_sigmoid_backward(num_test: int = 1000):
    for _ in range(num_test):
        dim = random.randint(3, 20)
        num_samples = random.randint(10, 100)
        x = torch.randn(num_samples, dim) * 2
        x = x - torch.mean(x, dim=0)
        
        x_np = x.numpy()
        x.requires_grad = True

        custom_sigmoid = SigmoidLayer()
        custom_y = custom_sigmoid(x_np)

        loss_obj = SumLoss()
        loss = loss_obj(nn.Sigmoid()(x))
        loss.backward()
        torch_grad = x.grad

        initial_upstream_grad = np.ones(shape=custom_y.shape) # np.sign(custom_y)
        custom_grad = custom_sigmoid.grad(x_np, initial_upstream_grad)

        assert np.allclose(custom_grad, torch_grad, atol=10 ** -6), "Please make sure the grad of Relu works correctly"

        x = torch.randn(num_samples, dim) * 2
        x = x - torch.mean(x, dim=0)

        x_np = x.numpy()
        x.requires_grad = True

        custom_relu = SigmoidLayer()
        custom_y = custom_relu(x_np)

        loss_obj = AbsSumLoss()
        loss = loss_obj(nn.Sigmoid()(x))
        loss.backward()
        torch_grad = x.grad

        initial_upstream_grad = np.sign(custom_y)
        custom_grad = custom_relu.grad(x_np, initial_upstream_grad)

        assert np.allclose(custom_grad, torch_grad, atol=10 ** -6), "Please make sure the grad of Relu works correctly"

def test_flatten_backward(num_test: int = 1000):
    for i in range(num_test):
        num_dims = np.random.randint(3, 10)
        shape = tuple([np.random.randint(3, 10) for _ in range(num_dims)])
        x = torch.randn(*shape)
        x_np = x.numpy()
        x.requires_grad = True

        custom_flatten = FlattenLayer()
        custom_y = custom_flatten(x_np)

        loss_obj = AbsSumLoss()
        loss = loss_obj(nn.Flatten()(x))
        loss.backward()
        torch_grad = x.grad

        initial_upstream_grad = np.sign(custom_y)
        custom_grad = custom_flatten.grad(x_np, initial_upstream_grad)

        assert np.allclose(custom_grad, torch_grad, atol=10 ** -6), "Please make sure the grad of Relu works correctly"
        
if __name__ == '__main__':
    # test relu
    
    test_relu_forward()
    test_relu_backward()
    
    # test softmax
    
    test_soft_forward()
    test_soft_backward()
    
    # test sigmoid

    test_sigmoid_forward()
    test_sigmoid_backward()

    test_flatten_backward()