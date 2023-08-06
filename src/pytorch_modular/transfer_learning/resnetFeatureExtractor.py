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

from typing import List, Union, Iterator
from pathlib import Path
from collections import OrderedDict

from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
# Bottleneck is the class that contains the Residual block
from torchvision.models.resnet import Bottleneck
from torch.utils.data import DataLoader
import torchvision.transforms as tr

from src.pytorch_modular.transfer_learning.classification_head import ExponentialClassifierHead
from src.pytorch_modular.directories_and_files import process_save_path
from src.pytorch_modular.data_loaders import create_dataloaders

HOME = os.getcwd()
sys.path.append(HOME)
sys.path.append(os.path.join(HOME, 'src'))

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
class RestNetFeatureExtractor(nn.Module):
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
    # TODO: THOROUGHLY TEST THIS METHOD
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

    def __init__(self,
                 num_blocks: int,  # the number of blocks to keep
                 blocks_type: str = LAYER_BLOCK,  # the type of blocks to consider (layers or residual blocks)
                 freeze: bool = True,  # whether to freeze the chosen layers or not
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        # first make sure the blocks_type argument is correct
        assert blocks_type in [LAYER_BLOCK, RESIDUAL_BLOCK], "MAKE SURE TO PASS A SUPPORTED TYPE OF BLOCKS"

        self.num_blocks = num_blocks
        # make sure to explicitly
        self.__net = resnet50(ResNet50_Weights.DEFAULT)
        # before proceeding to extract the feature set freeze the layer if needed
        if freeze:
            for para in self.__net.parameters():
                para.requires_grad = False

        self.feature_extractor = None
        self.modules = None
        self.feature_extractor = self.__feature_extractor_layers(self.num_blocks) \
            if blocks_type == LAYER_BLOCK else self.__feature_extractor_residuals(self.num_blocks)

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


"""
building a feature extract from a given pretrained model (resnet in this script) is simply the first step.
The most important step is to derive which layers to transfer to the downstream task 
"""


class ResnetFeatureSelector:
    """
    this class simulates (on a smaller scale) the paper's main experience: finding the appropriate number of
    layers to transfer to the target network: the network responsible for solving the downstream task.
    """

    def __init__(self,
                 classifier_head: nn.Module,
                 options: List[int] = None,
                 block_type: str = LAYER_BLOCK,
                 freeze: bool = True):

        if options is not None:
            options = [num for num in options if num <= 4]

        # if 'options' is not provided, use all the possible values
        if options is None:
            options = list(range(1, 5))

        if classifier_head is not None:
            if not hasattr(classifier_head, 'in_features'):
                raise TypeError("THE PASSED CLASSIFIER MUST HAVE THE A FIELD 'in_features' TO SET THE NUMBER OF "
                                "INPUT UNITS. SUCH NUMBER DEPENDS ON THE FEATURE EXTRACTOR")

        # the classifier head is
        self.head = classifier_head
        self.block_type = block_type
        self.freeze = freeze
        self.options = options
        self.fes = [ResnetFeatureSelector(num_blocks=o,
                                          block_type=self.block_type,
                                          freeze=self.freeze) for o in self.options]

    def experimental_setup(self,
                           train_data: Union[DataLoader, str, Path],
                           val_data: Union[DataLoader, str, Path],
                           train_transform: tr = None,
                           val_transform: tr = None,
                           num_classes: int = None):

        # train data represents the source of the training data
        # make sure that the val and train data come from similar sources: either directories or data loaders

        loaders_data = isinstance(train_data, DataLoader) and isinstance(val_data, DataLoader)
        path_data = isinstance(train_data, (Path, str)) and isinstance(val_data, (Path, str))
        if not (loaders_data or path_data):
            raise TypeError("BOTH TRAINING AND VALIDATION DATA SOURCE MUST BE SIMILAR:\n"
                            "EITHER BOTH DATALOADERS OR BOTH PATHS TO DIRECTORIES")

        if path_data:
            # make sure to process the paths
            val_data = process_save_path(val_data, file_ok=False)
            train_data = process_save_path(train_data, file_ok=False)
            # create the dataloaders
            train_data, val_data, num_classes = create_dataloaders(train_data,
                                                                   train_transform,
                                                                   val_data,
                                                                   val_transform)

        else:
            # if the data is given as loader, then the 'num_classes' must be set
            if loaders_data and num_classes is None:
                raise TypeError("IF THE DATA IS PROVIDED AS DATALOADER, THE NUMBER OF CLASSES"
                                "MUST BE EXPLICITLY SET")


if __name__ == '__main__':
    relu = nn.ReLU()
    i = relu.children()

