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
from tinyBackProp.activation_layers import SoftmaxLayer, ReLULayer
from tinyBackProp.losses import CrossEntropyLoss
from tinyBackProp.network import Network


torch.manual_seed(69)


def test_model_1(num_test: int = 10):
    for _ in range(num_test):
        # define the data
        num_classes = random.randint(3, 20)
        num_samples = random.randint(10, 100)
        x = torch.randn(num_samples, num_classes) * 2
        x_np = x.numpy()

        # define the target
        y_np = np.array([random.randint(0, 9) for _ in range(num_samples)])
        y = torch.from_numpy(y_np)

        # define torch model
        torch_model = nn.Linear(in_features=num_classes, out_features=10, bias=False)
        w = torch_model.weight.detach().numpy()
        torch_loss_obj = nn.CrossEntropyLoss()
        torch_predictions = torch_model.forward(x)
        torch_loss = torch_loss_obj(torch_predictions, y)

        torch_loss.backward()

        torch_grad = torch_model.weight.grad.detach().numpy()

        # define custom model
        linear, soft = LinearLayer(in_features=num_classes, out_features=10, weight_matrix=w.T), SoftmaxLayer()
        cle = CrossEntropyLoss(num_classes=10)

        net = Network([linear, soft])

        y_pred = net.forward(x_np)

        custom_loss = cle(y_pred, y_true=y_np)

        # make sure the losses are close
        assert abs(custom_loss - torch_loss.item()) <= 10 ** -5, "The losses are not the same"

        loss_upstream_grad = cle.grad(y_pred, y_np)

        # pass this to the model
        custom_grads = net.backward(loss_upstream_grad, learning_rate=0.01)

        g = custom_grads[-1].T

        assert np.allclose(g, torch_grad, atol=10 ** -5), "The gradients of the model are not the same"


def test_model_2(num_test: int = 10):
    for _ in range(num_test):
        # define the data
        h_dim = random.randint(20, 30)
        num_samples = random.randint(10, 100)
        x = torch.randn(num_samples, h_dim) * 2
        x_np = x.numpy()

        # define the target
        y_np = np.array([random.randint(0, 9) for _ in range(num_samples)])
        y = torch.from_numpy(y_np)

        # define torch model
        torch_l1 = nn.Linear(in_features=h_dim, out_features=20, bias=False)
        torch_l2 = nn.Linear(in_features=20, out_features=10, bias=False)

        w1 = torch_l1.weight.detach().numpy()
        w2 = torch_l2.weight.detach().numpy()

        torch_model = nn.Sequential(torch_l1, nn.ReLU(), torch_l2)
        torch_loss_obj = nn.CrossEntropyLoss()
        torch_predictions = torch_model.forward(x)
        torch_loss = torch_loss_obj(torch_predictions, y)

        torch_loss.backward()

        torch_grad1 = torch_l1.weight.grad
        torch_grad2 = torch_l2.weight.grad

        # define custom model
        l1, relu, l2, soft = (LinearLayer(in_features=h_dim, out_features=20, weight_matrix=w1.T),
                              ReLULayer(),
                              LinearLayer(in_features=20, out_features=10, weight_matrix=w2.T),
                              SoftmaxLayer())
        cle = CrossEntropyLoss(num_classes=10)

        net = Network([l1, relu, l2, soft])

        y_pred = net.forward(x_np)

        custom_loss = cle(y_pred, y_true=y_np)

        # make sure the losses are close
        assert abs(custom_loss - torch_loss.item()) <= 10 ** -5, "The losses are not the same"

        loss_upstream_grad = cle.grad(y_pred, y_np)

        # pass this to the model
        custom_grads = net.backward(loss_upstream_grad, learning_rate=0.01)

        g1, g2 = custom_grads[-1].T, custom_grads[-3].T


        assert torch_grad1 != w1 and torch_grad2 != w2
        # make sure the first gradient is correct
        assert np.allclose(g1, torch_grad1, atol=10 ** -6), "The gradients of the model are not the same"

        # make sure the second gradient is correct
        assert np.allclose(g2, torch_grad2, atol=10 ** -6), "The gradients of the model are not the same"


if __name__ == '__main__':
    test_model_2()
