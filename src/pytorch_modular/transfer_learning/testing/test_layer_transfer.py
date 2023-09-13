"""
This script contains functionality to test the forward pass of the ResNetFeatureExtractor class
"""
import random
import os
import sys
import numpy as np
import torch
import unittest

# since the 'src' directory is a non-direct parent of the current file
# it is not visible which might raise a ModuleNotFoundError

# this small piece of code overcomes this

from pathlib import Path

try:
    import src.pytorch_modular.transfer_learning.resnetFeatureExtractor as tl_res
except ModuleNotFoundError:
    h = os.getcwd()
    while 'src' not in os.listdir(h):
        print(h)
        h = Path(h).parent
    # at this point 'dir' should be the root of the repository
    # make sure to convert the path to a string as sys.path does not recognize Path objects
    sys.path.append(str(h))

import src.pytorch_modular.transfer_learning.resnetFeatureExtractor as tl_res
from src.pytorch_modular.pytorch_utilities import get_default_device
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from collections import OrderedDict

# set the random seed for reproducibility
random.seed(69)


class FeatureExtractorTest(nn.Module):
    """
    This class is a simple but non-expandable implementation of the featureExtractor class
    Having close outputs for both classes guarantees the correctness of the ResNetFeatureExtractor class\
    """

    def __init__(self, num_layers: int,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__net = resnet50(ResNet50_Weights.DEFAULT)
        for p in self.__net.parameters():
            p.requires_grad = False
        self.num_layers = num_layers
        self.layers_dict = OrderedDict([(1, self.__net.layer1),
                                        (2, self.__net.layer2),
                                        (3, self.__net.layer3),
                                        (4, self.__net.layer4)])
        self.modules = [self.__net.conv1, self.__net.bn1, self.__net.relu, self.__net.maxpool]
        self.modules.extend([self.layers_dict[i] for i in range(1, min(self.num_layers, 4) + 1)])
        self.modules.append(self.__net.avgpool)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # this function is basically a copy of the ResNet50._forward_impl function
        # https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

        x = self.__net.conv1(x)
        x = self.__net.bn1(x)
        x = self.__net.relu(x)
        x = self.__net.maxpool(x)

        for i in range(1, min(self.num_layers, 4) + 1):
            x = self.layers_dict[i].forward(x)
        # make sure to pass through the average pooling layer at the end
        x = self.__net.avgpool(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


class TestFeatureExtractorOutput(unittest.TestCase):
    def test_total_output(self):
        # calling this function will
        device = get_default_device()
        layers = list(range(1, 5))
        # add a random number larger 4 to make sure the ResNetFeatureExtractor code doesn't break
        layers.append(random.randint(a=5, b=10))

        for v in layers:
            # first create the needed modules
            test_fe = FeatureExtractorTest(num_layers=v)
            tested_fe = tl_res.ResNetFeatureExtractor(num_blocks=v)

            tested_fe.to(device)
            test_fe.to(device)

            # make sure to set both modules to test modes
            test_fe.eval()
            tested_fe.eval()

            for _ in range(1000):
                batch, height, width = random.randint(a=4, b=12), \
                    random.randint(a=244, b=512), \
                    random.randint(a=244, b=512)

                # torch.conv2d only works with 3 channel input
                x = torch.rand(batch, 3, height, width).to(device)
                true_output = test_fe(x).cpu().numpy()
                test_output = tested_fe(x).cpu().numpy()

                correct = np.allclose(true_output, test_output, equal_nan=True)
                self.assertTrue(correct, 'The outputs of both classes are too different')


if __name__ == '__main__':
    unittest.main()
