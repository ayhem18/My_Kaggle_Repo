"""
This script contains functionalities to build classifiers on top of the pretrained model
'RESNET 50' provided by pytorch module

This script is mainly inspired by this paper: 'https://arxiv.org/abs/1411.1792'
as they suggest an experimental framework to find the most transferable / general layers
in pretrained network. I am applying the same framework on the resnet architecture.
"""
import os
import sys
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
# Bottleneck is the class that contains the Residual block
from torchvision.models.resnet import Bottleneck
from collections import OrderedDict

HOME = os.getcwd()
sys.path.append(HOME)
sys.path.append(os.path.join(HOME, 'src'))

# from torchvision.models.resnet50 import
LAYER_BLOCK = 'layer'
RESIDUAL_BLOCK = 'residual'


# let's create a utility function real quick
def contains_fc_layer(module: nn.Module) -> bool:
    """
    This function returns whether the module contains a Fully connected layer or not
    """
    m = isinstance(module, nn.Linear)
    sub_m = any([isinstance(m, nn.Linear) for m in module.modules()])
    return m and sub_m


# noinspection PyUnresolvedReferences,PyShadowingNames
class FeatureExtractor(nn.Module):
    # the model's architecture refers to blocks with the same number of channels as 'layers'
    # a function to build the transferred part from the original resnet model
    # in case of 'layer' blocks

    def __feature_extractor_layers(self, number_of_layers: int):
        modules_generator = self.__net.named_children()
        # the modules will be saved with their original names in an OrderedDict and later merged
        # into a single nn.Module using nn.Sequential
        modules_to_keep = []
        counter = 0

        for name, module in modules_generator:
            # we will only consider non fully connected components of the network
            if not contains_fc_layer(module):
                if 'layer' not in name or counter < number_of_layers:
                    modules_to_keep.append((name, module))
                    # only increment counter if the name contains 'layer': meaning it is a layer block
                    counter += int('layer' in name)

        self.modules = [m for n, m in modules_to_keep]
        fe = nn.Sequential(OrderedDict(modules_to_keep))
        return fe

    # a function to build the transferred part from the original resnet model
    # is case of 'residual' block
    def __feature_extractor_residuals(self, number_of_blocks: int):
        modules_generator = self.__net.named_children()
        counter = 0
        modules_to_keep = []
        # this part is slightly more complex
        for name, module in modules_generator:
            # discard any fully connected components of the network
            if not contains_fc_layer(module):
                # the residual blocks lie within the 'layer' blocks
                if 'layer' in name:
                    # here we will build a nn.Sequential object that uses as many residual blocks as possible
                    new_layer = []
                    for _, sub_module in module.named_children():
                        if not isinstance(sub_module, Bottleneck) or counter < number_of_blocks:
                            new_layer.append(sub_module)
                            # only increment the counter is the sub_module added is
                            counter += int(isinstance(sub_module, BottleNeck))
                    # time to wrap this new layer in a nn.Sequential to use freely with the rest of the model
                    new_layer = nn.Sequential(new_layer)
                    modules_to_keep.append(new_layer)
                else:
                    modules_to_keep.append(module)

        self.modules = [m for n, m in modules_to_keep]
        fe = nn.Sequential(modules_to_keep)
        return fe

    def __init__(self, blocks_to_keep: int,  # the number of blocks to keep
                 blocks_type: str = LAYER_BLOCK,  # the type of blocks to consider (layers or residual blocks)
                 freeze: bool = True,  # whether to freeze the chosen layers or not
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        # first make sure the blocks_type argument is correct
        assert blocks_type in [LAYER_BLOCK, RESIDUAL_BLOCK], "MAKE SURE TO PASS A SUPPORTED TYPE OF BLOCKS"

        self.num_blocks = blocks_to_keep
        # make sure to have
        self.__net = resnet50(ResNet50_Weights.DEFAULT)
        # before proceeding to extract the feature set freeze the layer if needed
        if freeze:
            for para in self.__net.parameters():
                para.requires_grad = False

        self.feature_extractor = None
        self.modules = None
        self.feature_extractor = self.__feature_extractor_layers(blocks_to_keep) \
            if blocks_type == LAYER_BLOCK else self.__feature_extractor_residuals(blocks_to_keep)

    def forward(self, x: torch.Tensor):
        # the forward function in the ResNet class simply calls the forward function
        # of each submodule consecutively: which is equivalent to saving all modules in a nn.Sequential module
        # and calling the forward function of the component.
        return self.feature_extractor.forward(x)

    def __str__(self):
        # the default __str__ function will display the self.__net module as well
        # which might be confusing as .__net is definitely not part of the forward pass of the model
        return self.feature_extractor.__str__()


if __name__ == '__main__':
    default_weights = ResNet50_Weights.DEFAULT
    net = resnet50(default_weights)
    # let's build a feature extractor
    print(net)