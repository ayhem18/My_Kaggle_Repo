"""
This script contains functionalities to build classifiers on top of the pretrained model
'RESNET 50' provided by pytorch module

This script is mainly inspired by this paper: 'https://arxiv.org/abs/1411.1792'
as they suggest an experimental framework to find the most transferable / general layers
in pretrained network. I am applying the same framework on the resnet architecture.
"""

import os
import sys
import warnings

import torch
import torchvision.transforms as tr

from typing import List, Union, Iterator, Dict
from pathlib import Path
from collections import OrderedDict
from _collections_abc import Sequence

from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
# Bottleneck is the class that contains the Residual block
from torchvision.models.resnet import Bottleneck
from torch.utils.data import DataLoader

try:
    from src.pytorch_modular.pytorch_utilities import get_default_device
except ModuleNotFoundError:
    h = os.getcwd()
    if 'src' not in os.listdir(h):
        # it means HOME represents the script's parent directory
        while 'src' not in os.listdir(h):
            h = Path(h).parent

    sys.path.append(str(h))

from src.pytorch_modular.directories_and_files import process_save_path
from src.pytorch_modular.data_loaders import create_dataloaders
from src.pytorch_modular.image_classification import classification_head as ch
from src.pytorch_modular.dimensions_analysis import dimension_analyser as da

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
building a feature extractor from a given pretrained model (resnet in this script) is simply the first step.
The most important step is to derive which layers to transfer to the downstream task 
"""


class ResnetFeatureSelector:
    """
    this class simulates (on a smaller scale) the paper's main experience: finding the appropriate number of
    layers to transfer to the target network: the network responsible for solving the downstream task.
    """

    @classmethod
    def _experiment_data_setup(cls,
                               train_data: Union[DataLoader, str, Path],
                               val_data: Union[DataLoader, str, Path],
                               train_transform: tr = None,
                               val_transform: tr = None,
                               num_classes: int = None) -> tuple[DataLoader, DataLoader, int]:
        """
        This function prepares the data needed for the experiment. The funtion is made static
        as features extractors are not invoked at this stage

        Arguments:
            train_data: can be either a path to a directory or a dataloader
            val_data: either a path to a directory or a dataloader
            train_transform: only considered if the training data is a path
            val_transform: only considered if the val data is a path
            num_classes: required if the data is passed as DataLoaders
        """

        # first let's check the types
        if not isinstance(train_data, (str, Path, DataLoader)) or not isinstance(val_data, (str, Path, DataLoader)):
            raise TypeError(f"Please make sure to pass the training data either as:\n 1. a path: {(str, Path)} \n"
                            f"2. a dataloader: {DataLoader}\n"
                            f"training data's type :{type(train_data)}\n"
                            f"validation data's type: {type(val_data)}")

        # make sure that both training and validation data are of the same type
        path_data = isinstance(train_data, (str, Path)) and isinstance(val_data, (str, Path))
        load_data = isinstance(train_data, DataLoader) and isinstance(val_data, DataLoader)

        if not (path_data or load_data):
            raise TypeError("The training and validation data must be of the same source")

        # if the data is from a path
        # create dataloaders
        if path_data:
            train_data = process_save_path(train_data, file_ok=False)
            val_data = process_save_path(val_data, file_ok=False)
            train_dataloader, val_dataloader, num_classes = create_dataloaders(train_data,
                                                                               train_transform,
                                                                               val_data,
                                                                               val_transform)

            return train_dataloader, val_dataloader, num_classes

        # at this point is it known the data was passed as dataloaders
        if load_data and num_classes is None:
            raise TypeError("IF THE DATA IS PROVIDED AS DATALOADER, the `num_classes` argument must be"
                            "explicitly set.")

        if train_transform is not None or val_transform is not None:
            # raise a warning if train_transform or val_transform is set
            warnings.warn("At least One transformation was explicitly set. Transformations are ignored"
                          "since dataloaders were passed as data sources.")

        return train_data, val_data, num_classes

    def __init__(self,
                 classifiers: List[nn.Module] = None,
                 options: List[int] = None,
                 block_type: str = LAYER_BLOCK,
                 freeze: bool = True):

        if options is not None:
            if not isinstance(options, Sequence) or len(options) == 0:
                raise ValueError("The `options` argument is expected to a non-empty iterable\nFound an object "
                                 f"of type {type(options)} "
                                 f"{f'of length {len(options)}.' if isinstance(options, Sequence) else ''}")

            for element in options:
                if not isinstance(element, int):
                    raise TypeError("The `options` argument is expected to an iterable of integers")
                options = [num for num in options if num <= 4]

        # if 'options' is not explicitly set, use all the possible values
        if options is None:
            options = list(range(1, 5))

        if classifiers is not None:
            for c in classifiers:
                if not (hasattr(c, 'in_features') and hasattr(c, 'num_classes')):
                    raise TypeError("Classifiers are expected to have A field `in_features` "
                                    "and `num_classes` to set the number of "
                                    "input units as well as output units. "
                                    "Such numbers depends on the feature extractor and the training data")

        # the classifier head is
        self.classifiers = classifiers
        self.block_type = block_type
        self.freeze = freeze
        self.options = options
        self.fes = [RestNetFeatureExtractor(num_blocks=o,
                                            block_type=self.block_type,
                                            freeze=self.freeze)
                    for o in self.options]

        # a field used for saving the complete networks
        self.networks = [None for _ in self.options]

    def _build_networks(self,
                        train_data: DataLoader,
                        num_classes: int):

        if self.classifiers is None:
            # create as many classifiers as options
            self.classifiers = [ch.GenericClassifier(None, num_classes=num_classes) for _ in self.options]

        # dimension analyser
        dim_analyser = da.DimensionsAnalyser(method='static')
        # extracts the input shape from the dataloader
        input_shape = dim_analyser.analyse_dimensions_dataloader(train_data)

        for index, feature_extractor in enumerate(self.fes):
            # first extract the output dimensions from the feature extractor
            fe_output_shape = dim_analyser.analyse_dimensions(input_shape, feature_extractor)
            # time to compute the number of input units fed into the classifier
            flatten_output = dim_analyser.analyse_dimensions(fe_output_shape, nn.Flatten())

            assert len(flatten_output) == 2, "the output shape of the flatten layer is incorrect"

            batch_size, input_units = flatten_output

            # set the input_units of the classifier
            self.classifiers[index].in_features = input_units

            # check the value of the `num_classes` field as setting it might be computationally expensive
            if self.classifiers[index].num_classes != num_classes:
                self.classifiers[index].num_classes = num_classes

            # now time to finally put the pieces together
            self.networks[index] = nn.Sequential(feature_extractor,
                                                 nn.Flatten(),
                                                 self.classifiers[index])

    def _experiment_core(self,
                         train_data: DataLoader,
                         val_data: DataLoader,
                         train_configuration: Dict
                         ):
        # extract all the parameters from the train_configuration
        pass


if __name__ == '__main__':
    relu = nn.ReLU()
    i = relu.children()
