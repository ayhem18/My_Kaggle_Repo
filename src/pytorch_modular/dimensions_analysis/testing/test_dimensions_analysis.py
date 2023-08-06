"""
This script contains functionalities to test the code written in both
'layer specific' and 'dimension_analyser' scripts
"""

import random
import os
import sys
import numpy as np
import torch
import unittest

from pathlib import Path
from random import randint as ri

try:
    import src.pytorch_modular.random_generation.random_components as rc
except ModuleNotFoundError:
    h = os.getcwd()
    while 'src' not in os.listdir(h):
        print(h)
        h = Path(h).parent
    # at this point 'dir' should be the root of the repository
    # make sure to convert the path to a string as sys.path does not recognize Path objects
    sys.path.append(str(h))

import src.pytorch_modular.random_generation.random_components as rc
import src.pytorch_modular.dimensions_analysis.dimension_analyser as da
from src.pytorch_modular.pytorch_utilities import get_default_device


class TestOutputAnalysis(unittest.TestCase):
    def test_output_analysis(self):
        # using the default device
        device = get_default_device()
        # create a module analyser
        # make sure to use static
        analyser = da.DimensionsAnalyser(method=da._STATIC)
        # 1000 test cases should be good enough
        for _ in range(1000):
            in_channels, conv_block = rc.random_conv_block(return_in_c=True)
            conv_block.to(device)
            conv_block.eval()
            # create the input shape
            random_batch = ri(1, 10)    # keeping the random_batch small for memory concerns (mainly with GPU)x
            random_height = ri(1000, 2000)
            random_width = ri(1000, 2000)

            input_shape = (random_batch, in_channels, random_height, random_width)
            computed_output_shape = analyser.analyse_dimensions(input_shape, conv_block)

            input_tensor = torch.ones(size=input_shape).to(device)
            output_tensor = conv_block.forward(input_tensor)

            self.assertEquals(tuple(output_tensor.size), computed_output_shape), "THE OUTPUT SHAPES ARE DIFFERENT"


if __name__ == '__main__':
    unittest.main()
