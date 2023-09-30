"""
This script contains functionalities to build a feature extractor based on the pretrained DenseNet model
"""

import torch

from typing import Iterator, Tuple
from collections import OrderedDict

from torch import nn
from torch.nn.modules.module import Module
from torchvision.models import densenet121, DenseNet121_Weights


def contains_fc_layer(module: nn.Module) -> bool:
    """
    This function returns whether the module contains a Fully connected layer or not
    """
    sub_m = any([isinstance(m, nn.Linear) for m in module.modules()])
    m = isinstance(module, nn.Linear)
    return m and sub_m


class DenseNetFeatureExtractor(nn.Module):
    default_transform = DenseNet121_Weights.DEFAULT.transforms()

    # the major part of the architecture is the DenseBlock
    def __feature_extractor_blocks(self,
                                   num_blocks: int,
                                   minimal: bool = True) -> nn.Sequential:
        # the dense121 children generator contains only 2 children: The feature extractor and the final linear layer

        modules_generator = list(self.__net.named_children())[0][-1].named_children()
        # modules_generator = self.__net
        # modules_generator = self.__net.named_children()
        # the modules will be saved with their original names in an OrderedDict and later merged
        # into a single nn.Module using nn.Sequential
        modules_to_keep = []
        dense_counter = 0
        transition_counter = 0
        for name, module in modules_generator:
            # we will only consider non fully connected components of the network
            if not contains_fc_layer(module):
                if 'dense' in name:
                    # check the counter
                    if dense_counter < num_blocks:
                        modules_to_keep.append((name, module))
                        dense_counter += 1
                    continue

                if 'transition' in name:
                    # check the counter
                    if transition_counter < num_blocks:
                        modules_to_keep.append((name, module))
                        transition_counter += 1
                    continue

                # reaching this part of the code means, the layer is neither a 'dense' or a 'transition' block:
                # first check if all transition and dense block have been already added                
                if transition_counter == dense_counter == num_blocks:
                    if not minimal:
                        modules_to_keep.append((name, module))
                    continue

                # this code will be reached for modules before the first dense block
                modules_to_keep.append((name, module))

        self.modules = [m for _, m in modules_to_keep]
        fe = nn.Sequential(OrderedDict(modules_to_keep))
        return fe

    def __init__(self,
                 num_blocks: int,  # the number of blocks to keep
                 minimal: bool = True,
                 freeze: bool = True,  # whether to freeze the chosen layers or not
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_blocks = num_blocks
        # make sure to explicitly
        self.__net = densenet121(weights=DenseNet121_Weights.DEFAULT)

        # freeze the layers if needed
        if freeze:
            for para in self.__net.parameters():
                para.requires_grad = False

        self.feature_extractor = self.__feature_extractor_blocks(self.num_blocks, minimal)

    def forward(self, x: torch.Tensor):
        # the forward function in the ResNet class simply calls the forward function
        # of each submodule consecutively: which is equivalent to saving all modules in a nn.Sequential module
        # and calling the forward method.
        return self.feature_extractor.forward(x)

    def __str__(self):
        # the default __str__ function will display the self.__net module as well
        # which might be confusing as .__net is definitely not part of the forward pass of the model
        return self.feature_extractor.__str__()

    def children(self) -> Iterator['Module']:
        # not overloading this method will return to an iterator with 2 elements: self.__net and self.feature_extractor
        return self.feature_extractor.children()

    def modules(self) -> Iterator[nn.Module]:
        return self.feature_extractor.modules()

    def named_children(self) -> Iterator[Tuple[str, Module]]:
        return self.feature_extractor.named_children()