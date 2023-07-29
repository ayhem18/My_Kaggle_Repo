"""
This script contains functionality to test the forward pass of the FeatureExtractor class
"""
import random

import numpy as np
import torch
import src.pytorch_modular.transfer_learning.transfer_resnet as tl_res
import unittest

from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from collections import OrderedDict

# set the random seed for reproducibility
random.seed(69)


class FeatureExtractorTest(nn.Module):
    """
    This class is a simple but non-expandable implementation of the featureExtractor class
    extremly close outputs of both classes guarantees the correctness of FeatureExtractor class
    """

    def __init__(self, num_layers: int,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__net = resnet50(ResNet50_Weights.DEFAULT)
        self.num_layers = num_layers
        self.layers_dict = OrderedDict([(1, self.__net.layer1),
                                        (2, self.__net.layer2),
                                        (3, self.__net.layer3),
                                        (4, self.__net.layer4)])

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # this function is basically a copy of the ResNet50._forward_impl function
        # https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

        x = self.__net.conv1(x)
        x = self.__net.bn1(x)
        x = self.__net.relu(x)
        x = self.__net.maxpool(x)

        for i in range(1, min(self.num_layers, 4) + 1):
            x = self.layers_dict[i].forward(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._forward_impl(x)


class TestFeatureExtractorOutput(unittest.TestCase):
    def __init__(self, num_inner_tests: int = 1000):
        super().__init__()
        self.tests = num_inner_tests

    def test_total_output(self):
        layers = list(range(1, 5))
        # add a random number larger 4 to make sure the FeatureExtractor code doesn't break
        layers.append(random.randint(a=5, b=10))
        for v in layers:
            # first create the needed modules
            test_fe = FeatureExtractorTest(num_layers=v)
            tested_fe = tl_res.FeatureExtractor(blocks_to_keep=v)
            # make sure to set both modules to test modes
            test_fe.eval()
            tested_fe.eval()

            for _ in range(self.tests):
                batch, height, width = random.randint(a=4, b=12), \
                    random.randint(a=244, b=512), \
                    random.randint(a=244, b=512)

                # torch.conv2d only works with 3 channel input
                x = torch.rand(batch, 3, height, width)
                true_output = test_fe(x).detach().cpu().numpy()
                test_output = tested_fe(x).detach().cpu().numpy()
                correct = np.allclose(true_output, test_output, equal_nan=True)
                self.assertTrue(correct)


if __name__ == '__main__':
    unittest.main()
