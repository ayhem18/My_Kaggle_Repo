"""
The script resnetFeatureExtractor provides a script that builds a feature extractor from the resnet pretrained model
Nevertheless, this is merely the 1st step as the ultimate goal is to decide which parts of the pretrained model should
be transferred to the downstream task
"""
import os
import warnings

import numpy as np

import pytorch_modular.image_classification.utilities as ut
import pytorch_modular.image_classification.engine_classification as cls

from torch import nn
from torchvision import transforms as tr
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Union, List, Dict, Any
from _collections_abc import Sequence
from copy import deepcopy

from pytorch_modular.directories_and_files import process_save_path
from pytorch_modular.data_loaders import create_dataloaders
from pytorch_modular.image_classification import classification_head as ch
from pytorch_modular.dimensions_analysis import dimension_analyser as da
from pytorch_modular.transfer_learning.resnetFeatureExtractor import ResNetFeatureExtractor
from pytorch_modular.pytorch_utilities import save_model

LAYER_BLOCK = 'layer'
BEST = 'best'
LAST = 'last'


class ResnetFeatureSelector:
    """
    this class simulates (on a smaller scale) the paper's main experience: finding the appropriate number of
    layers to transfer to the target network: the network responsible for solving the downstream task.
    """

    @classmethod
    def _experiment_data_setup(cls,
                               train_data: Union[DataLoader, str, Path],
                               val_data: Union[DataLoader, str, Path],
                               batch_size: int = 64,  # picking a small safe value as default
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
            train_dataloader, val_dataloader, classes = create_dataloaders(train_dir=train_data,
                                                                           test_dir=val_data,
                                                                           batch_size=batch_size,
                                                                           train_transform=train_transform,
                                                                           val_transform=val_transform)

            return train_dataloader, val_dataloader, classes

        # at this point is it known the data was passed as dataloaders
        if load_data and num_classes is None:
            raise TypeError("IF THE DATA IS PROVIDED AS DATALOADER, the `num_classes` argument must be"
                            "explicitly set.")

        if train_transform is not None or val_transform is not None:
            # raise a warning if train_transform or val_transform is set
            warnings.warn("At least One transformation was explicitly set. Transformations are ignored"
                          "since dataloaders were passed as data sources.")

        return train_data, val_data, num_classes

    @classmethod
    def _verify_class_input(cls,
                            classifier: nn.Module,
                            options: List[int] = None):

        # the 'options' argument represents the number of layer blocks
        # are to be transferred to the downstream task model: They are expected to be a sequence
        # of INTEGERS
        if options is not None:
            # make sure the input is an iterable
            if not isinstance(options, Sequence) or len(options) == 0:
                raise ValueError("The `options` argument is expected to a non-empty iterable\nFound an object "
                                 f"of type {type(options)} "
                                 f"{f'of length {len(options)}.' if isinstance(options, Sequence) else ''}")

            # make sure each element is an integer
            for element in options:
                if not isinstance(element, int):
                    raise TypeError(f"The `options` argument is expected to an iterable of integers\nFound element"
                                    f"of type {type(element)}")
                options = [num for num in options if num <= 4]

        # the passed classifiers: expected to be nn.Module
        # and with 2 important attributes: 'in_features' and 'num_classes'
        if classifier is not None:
            if not isinstance(classifier, nn.Module):
                raise TypeError(f'The classifier head is expected to be of type {nn.Module}\n'
                                f'Found {type(classifier)}')

            if not (hasattr(classifier, 'in_features') and hasattr(classifier, 'num_classes')):
                raise TypeError("Classifiers are expected to have fields `in_features` "
                                "and `num_classes` to set the number of "
                                "input units as well as output units. "
                                "Such values cannot be generalized as they are determined by the feature extractor"
                                "and the downstream task respectively")

    def __init__(self,
                 classifier: nn.Module = None,
                 options: List[int] = None,
                 block_type: str = LAYER_BLOCK,
                 freeze: bool = True):
        """
        Args:
            classifier: classifier head used on top of the different feature extractors
            options: the different options: represents the number of layer / residual blocks in a feature extractor
            block_type: residual or layer blocks
            freeze: whether to freeze the feature extractor or not
        """
        # the classifier head is
        self.classifier = classifier
        self.block_type = block_type
        self.freeze = freeze
        self.options = options if options is not None else list(range(1, 5))
        self.fes = [ResNetFeatureExtractor(num_blocks=o,
                                           blocks_type=self.block_type,
                                           freeze=self.freeze)
                    for o in self.options]

        # a field used for saving the complete networks
        self.networks = None

    def _build_networks(self,
                        train_data: DataLoader,
                        num_classes: int):
        """
        This function will build the final candidate models for the downstream task by combining 3 main components:
            1. a feature extractor
            2. nn.Flatten() layer
            3. the given classifier head

        Args:
            train_data:
            num_classes: number of classes in the downstream task

        Returns:
        """
        # dimension analyser
        dim_analyser = da.DimensionsAnalyser(method='static')
        # extracts the input shape from the dataloader
        input_shape = dim_analyser.analyse_dimensions_dataloader(train_data)

        self.networks = [None for _ in self.options]

        for index, feature_extractor in enumerate(self.fes):
            # first extract the output dimensions from the feature extractor
            fe_output_shape = dim_analyser.analyse_dimensions(input_shape, feature_extractor)
            # time to compute the number of input units fed into the classifier
            flatten_output = dim_analyser.analyse_dimensions(fe_output_shape, nn.Flatten())

            assert len(flatten_output) == 2, "the output shape of the flatten layer is incorrect"

            _, input_units = flatten_output

            if self.classifier is None:
                classifier_head = ch.GenericClassifier(in_features=input_units, num_classes=num_classes)
            else:
                classifier_head = deepcopy(self.classifier)
                classifier_head.in_features = input_units
                classifier_head.num_classes = num_classes

            result_network = nn.Sequential(feature_extractor, nn.Flatten(), classifier_head)
            # now time to finally put the pieces together
            self.networks[index] = result_network

    def _verify_experiment_input(self,
                                 learning_rates: Union[Sequence[float], float],
                                 schedulers_params: Union[Sequence[Dict[str, float]], Dict[str, float]],
                                 train_configuration: Dict[str, Any],
                                 train_transform: tr,
                                 val_transform: tr,
                                 selection_criterion: str,
                                 selection_value: str,
                                 selection_split: str
                                 ):
        # make sure the learning rates and schedulers are set to correct types
        if not isinstance(learning_rates, Sequence):
            learning_rates = [learning_rates for _ in self.options]

        if isinstance(learning_rates, Sequence) and len(learning_rates) != len(self.options):
            raise ValueError("Please make sure to pass either 1 learning rate used across all networks "
                             f"or pass a learning rate for each network.\nFound {len(learning_rates)} rates while"
                             f"expecting {len(self.options)}")

        if not isinstance(schedulers_params, Sequence):
            schedulers_params = [schedulers_params for _ in self.options]

        if isinstance(schedulers_params, Sequence) and len(schedulers_params) != len(self.options):
            raise ValueError("Please make sure to pass either 1 dictionary of learning scheduler parameters:"
                             f" used across all networks or pass a learning rate for each network."
                             f"\nFound {len(schedulers_params)} dictionary parameters while"
                             f"expecting {len(self.options)}")

        # the default val and train transform are the ones associated with the default Resnet50 weights
        train_transform = ResNetFeatureExtractor.default_transform if train_transform is None else train_transform
        val_transform = ResNetFeatureExtractor.default_transform if val_transform is None else val_transform

        if not callable(train_configuration[ut.OPTIMIZER]):
            raise TypeError("The class of the optimizer is expected, and not an instance of it, as the parameters"
                            f"are to be set internally for each candidate network\nFound: "
                            f"{type(train_configuration[ut.OPTIMIZER])}")

        # the first step is to make sure the selection_criterion argument is either 'train_loss', 'val_loss'
        # 'train_metric', 'val_metric' where 'metric' is one of the metric passed in the train_configuration
        if not ((selection_criterion in [ut.TRAIN_LOSS, ut.VAL_LOSS]) or (train_configuration[ut.METRICS].keys())):
            raise ValueError(f"The 'selection criterion' argument is expected to be either {ut.TRAIN_LOSS}, "
                             f"{ut.VAL_LOSS} or {train_configuration[ut.METRICS].keys()}")

        # make sure the selection_value is either 'best' or 'last'
        selection_value = selection_value.lower()
        if selection_value not in [BEST, LAST]:
            raise ValueError(f"'selection_value' argument is expected to be either {BEST} or {LAST}\nFound "
                             f"{selection_value}")

        # make sure the selection_split argument is either 'train' or 'val'
        selection_split = selection_split.lower()
        if selection_split not in ['train', 'val']:
            raise ValueError(f"'selection_split' argument is expected to be either 'train' or 'val'"
                             f"\nFound{selection_split}")

        # return the processed values
        return (learning_rates, schedulers_params, train_transform, val_transform,
                selection_criterion, selection_value, selection_split)

    def select_downstream_model(self,
                                learning_rates: Union[List[float], float],
                                schedulers_params: Union[List[Dict[str, float]], Dict[str, float]],
                                train_configuration: Dict[str, Any],
                                train_data: Union[DataLoader, str, Path],
                                val_data: Union[DataLoader, str, Path],
                                selection_criterion: str = ut.TRAIN_LOSS,
                                selection_value: str = 'best',
                                selection_split: str = 'train',
                                log_dir: Union[Path, str] = None,
                                batch_size: int = 64,  # picking a small safe value as default
                                train_transform: tr = None,
                                val_transform: tr = None,
                                num_classes: int = None) -> nn.Module:
        # process the data
        (learning_rates, schedulers_params, train_transform, val_transform,
         selection_criterion, selection_value, selection_split) = self._verify_experiment_input(
            learning_rates,
            schedulers_params,
            train_configuration,
            train_transform,
            val_transform,
            selection_criterion,
            selection_value,
            selection_split)

        # set up the data for the experiment
        train_loader, val_loader, classes_info = self._experiment_data_setup(train_data=train_data,
                                                                             val_data=val_data,
                                                                             batch_size=batch_size,
                                                                             train_transform=train_transform,
                                                                             val_transform=val_transform,
                                                                             num_classes=num_classes)
        num_classes = len(classes_info) if isinstance(classes_info, Dict) else classes_info

        # as the source of the training data is now available, we can build the different networks
        self._build_networks(train_data=train_loader, num_classes=num_classes)

        # before proceeding with training the different models, save the original callable object passed
        # as 'Optimizer' and 'LR Scheduler'

        optimizer_callable_obj = train_configuration[ut.OPTIMIZER]
        lr_scheduler_callable_obj = train_configuration[ut.SCHEDULER]

        parent_log_dir = process_save_path(log_dir)
        parent_log_dir = os.path.join(log_dir, f'experience_{len(os.listdir(log_dir)) + 1}')

        results = []
        for net, option, lr, lr_sc_params in zip(self.networks, self.options, learning_rates, schedulers_params):

            print("#" * 100)
            print(f"training network with option {option} started\n\n")

            # make sure to set the optimizer in the train configuration used the original callable object
            # saved before the training loop
            train_configuration[ut.OPTIMIZER] = optimizer_callable_obj(net.parameters(), lr=lr)

            train_configuration[ut.SCHEDULER] = lr_scheduler_callable_obj(train_configuration[ut.OPTIMIZER],
                                                                          **lr_sc_params)

            # save logs inside different sub folders of the 'parent_log_dir'
            network_log_dir = os.path.join(parent_log_dir, f'resnet_{option}_block',
                                           f'{"" if option == 1 else "s"}') if log_dir is not None else None

            performance_dict = cls.train_model(model=net,
                                               train_dataloader=train_loader,
                                               test_dataloader=val_loader,
                                               train_configuration=train_configuration,
                                               # make a different logging directory for each model trained
                                               log_dir=network_log_dir
                                               )
            # the choice of the model depends on the criterion chosen
            if selection_value == BEST:
                if selection_criterion in [ut.TRAIN_LOSS, ut.VAL_LOSS]:
                    results.append(performance_dict[f'best_{selection_criterion}'])
                else:
                    results.append(performance_dict[f'best_{selection_split}_{selection_criterion}'])

            else:  # consider the last value
                if selection_criterion in [ut.TRAIN_LOSS, ut.TEST_LOSS]:
                    results.append(performance_dict[f'{selection_criterion}'][-1])
                else:
                    results.append(performance_dict[f'{selection_split}_{selection_criterion}'][-1])

            print("#" * 100)
            print(f"training network with option {option} completed\n\n")

        # is the criterion is a loss, then the best model is the one with the lowest loss
        # if it is a metric, then the best model is the one with the largest value
        selected_model = None
        if selection_criterion in [ut.TRAIN_LOSS, ut.TEST_LOSS]:
            min_loss_index = np.argmin(results)
            selected_model = self.networks[min_loss_index]

        max_metric_index = np.argmax(results)
        selected_model = self.networks[max_metric_index]

        save_model_dir = os.path.join(parent_log_dir, 'selected_model')
        # save the selected_model
        save_model(selected_model, path=save_model_dir)

        return selected_model
