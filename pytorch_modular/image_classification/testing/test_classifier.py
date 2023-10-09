"""
This script is mainly designed to test the GenericClassifier class
"""
import os
import sys
import unittest
import torch

from random import randint as ri
from pathlib import Path

try:
    from src.pytorch_modular.pytorch_utilities import get_default_device
except ModuleNotFoundError:
    h = os.getcwd()
    if 'src' not in os.listdir(h):
        # it means HOME represents the script's parent directory
        while 'src' not in os.listdir(h):
            h = Path(h).parent

    sys.path.append(str(h))

from src.pytorch_modular.pytorch_utilities import get_default_device
from src.pytorch_modular.image_classification.classification_head import GenericClassifier


class TestGenericClassifier(unittest.TestCase):
    def test_forward_pass_random(self):
        device = get_default_device()
        for _ in range(1000):
            # random in_features
            in_feats = ri(100, 1000)
            num_classes = ri(1, 10)
            len_units = ri(0, 25)
            units = [ri(50, 100) for _ in range(len_units)]

            if num_classes < 2:
                with self.assertRaises(ValueError):
                    GenericClassifier(in_features=in_feats,
                                      num_classes=num_classes,
                                      hidden_units=units)
                continue

            head = GenericClassifier(in_features=in_feats,
                                     num_classes=num_classes,
                                     hidden_units=units).to(device)

            input_tensor = torch.ones(size=(ri(1, 5), in_feats)).to(device)
            output_tensor = head(input_tensor)

            # test the output
            output_dim = 1 if num_classes == 2 else num_classes
            self.assertEqual(output_dim, output_tensor.size()[1])

            # check the dimensions inside
            layers = list(head.children())
            self.assertEqual(len(layers), 2 * len(units) + 1)

    def test_forward_pass(self):
        device = get_default_device()
        in_feats = 4096
        out = 2
        units = [2048, 1024, 512, 128, 32]

        head = GenericClassifier(in_features=in_feats,
                                 num_classes=out,
                                 hidden_units=units).to(device)

        x = torch.ones(size=(ri(1, 5), in_feats)).to(device)
        output_tensor = head(x)

        self.assertEqual(output_tensor.size()[1], 1)
        units.append(1)

        for i, l in enumerate(head.children()):
            x = l(x)
            self.assertEqual(x.size()[1], units[i // 2])

    def test_setters(self):
        device = get_default_device()
        for _ in range(10):
            in_feats = 4096
            out = ri(3, 10)
            units = [1000, 500, 250]
            head = GenericClassifier(in_features=in_feats,
                                     num_classes=out,
                                     hidden_units=units).to(device)

            x = torch.ones(size=(ri(1, 5), in_feats)).to(device)
            head.forward(x)

            new_in = ri(256, 512)
            head.in_features = new_in
            head.to(device)

            with self.assertRaises(RuntimeError):
                head.forward(x.to(device))

            head.in_features = in_feats
            head.to(device)
            new_out = ri(3, 100)
            head.num_classes = new_out
            head.to(device)
            x = head.forward(x.to(device))
            self.assertEqual(x.size()[1], new_out)


if __name__ == '__main__':
    unittest.main()
