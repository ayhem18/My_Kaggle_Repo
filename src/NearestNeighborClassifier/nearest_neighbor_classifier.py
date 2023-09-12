"""
This script contains my implementation of the Nearest neighbor classifier
"""
import torch
import numpy as np

from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from _collections_abc import Sequence
from typing import Union, Tuple, List
from PIL import Image
from src.NearestNeighborClassifier import distances as dis


class NearestNeighborClassifier(ABC):
    _PIXEL_WISE_DIFF = 'pixel_wise'
    _MATRIX_MUL_DIFF = 'matrix_mul'
    _DISTANCES = [_PIXEL_WISE_DIFF, _MATRIX_MUL_DIFF]

    @classmethod
    def _to_numpy_list(cls, inputs: Union[Sequence[np.ndarray], np.ndarray, torch.Tensor]) -> List[np.ndarray]:
        if isinstance(inputs, torch.Tensor):
            if len(inputs.shape) < 3:
                raise ValueError("inputs is expected to be at least 3-dimensional: an image\n"
                                 f"Found input of shape: {tuple(inputs.shape)}")

            # if the input is of shape 3, we can assume it is a single image
            inputs = [inputs] if len(inputs.shape) == 3 else inputs
            return [ip.detach().cpu().permute((1, 2, 0)).numpy() for ip in inputs]

        if isinstance(inputs, np.ndarray):
            if len(inputs.shape) < 3:
                raise ValueError("inputs is expected to be at least 3-dimensional: an image\n"
                                 f"Found input of shape: {tuple(inputs.shape)}")

            # if the input is of shape 3, we can assume it is a single image
            inputs = [inputs] if len(inputs.shape) == 3 else inputs
            return [ip for ip in inputs]

        return inputs if isinstance(inputs, List) else list(inputs)

    def __init__(self, resize: Union[int, Tuple[int, int]] = None,
                 distance: str = 'pixel_wise') -> None:
        """
        Args:
            resize (Union[int, Tuple[int, int]], optional): _description_.
                Defaults to None: needed if the images are of different sizes
            distance (str, optional): _description_. Defaults to 'pixel_wise'.
            
        Raises:
            ValueError: if the distance passed is not Supported
        """
        if distance not in self._DISTANCES:
            raise ValueError(f"The distance is expected to be one of the supported distances: {self._DISTANCES}\n"
                             f"Found the distance: {distance}. Make sure to initialize the object with a supported "
                             f"distance")

        self._resize = resize if isinstance(resize, tuple) or resize is None else (resize, resize)
        self._distance = distance
        # boolean flag to instantly know whether the model was fitted to some training data.
        self._fitted = False

        self._train_data = None
        self._train_labels = None
        self._data_labels = None

    def _preprocess(self, X: Union[Sequence[np.ndarray], np.ndarray, torch.Tensor]):
        # the function is abstract, nevertheless the different subclasses still share the type preprocessing
        # and resize steps
        X = self._to_numpy_list(X)
        # resize the images
        if self._resize is not None:
            X = [np.asarray(Image.fromarray(x).resize(self._resize)) for x in X]
        return X

    @abstractmethod
    def fit(self, train_data: Union[Sequence[np.ndarray], np.ndarray, torch.Tensor],
            train_labels: List[Union[str, int]],
            iterator_like: bool = True):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class BasicNearestNeighborClassifier(NearestNeighborClassifier):
    def __init__(self, resize: Union[int, Tuple[int, int]] = None,
                 distance: str = 'pixel_wise'):
        super().__init__(resize, distance)

    def fit(self,
            train_data: Union[Sequence[np.ndarray], np.ndarray, torch.Tensor, DataLoader],
            train_labels: List[Union[str, int]] = None,
            iterator_like: bool = True) -> None:
        """

        Args:
            train_data:
            train_labels:
            iterator_like:

        Returns:
        """
        
        # make sure the labels are passed if the data is not iteartor_like
        if not iterator_like and train_labels is None:
            raise TypeError(f"The function expects either the argument 'iterator_like' to be {True}" 
                            f"or to have bothe labels and training data passed\nFound: labels as {train_labels}")
        
        if iterator_like:
            if not isinstance(train_data, DataLoader):
                raise TypeError(("if the argument 'iterator_like' is set to True\n"
                                 f"The training data is expected as a Pytorch DataLoader\nFound: {type(train_data)}"))
            self._train_data = train_data
            self._fitted = True
            return

        train_data = super()._preprocess(train_data)
        # the idea here is just to save the images and their labels
        self._train_data = train_data
        self._train_labels = train_labels
        self._data_labels = [(x, y) for (x, y) in zip(self._train_data, self._train_labels)]
        # set self.fitted to True
        self._fitted = True

    def _predict(self, test_image: np.ndarray) -> Union[str, int]:
        test_image = self._preprocess(test_image)
        # the predict function behavior depends mainly on the nature of the training data,
        # in other words depending on the value of the self.combined
        if self._data_labels is None:
            # this means the data is iterator like and the predictions should be made in batches
            final_label = None
            for _, (data, labels) in enumerate(self._train_data):
                # make sure to process the data
                data = self._preprocess(data)
                combined = [(x, y) for (x, y) in zip(data, labels)]
                min_distance = float('inf')


                if self._distance == self._PIXEL_WISE_DIFF:
                    batch_nearest, label = min(combined,
                                            key=lambda combined_input: 
                                            dis.pixel_wise_difference(combined_input[0], test_image))
                    
                    current_min = dis.pixel_wise_difference(test_image, batch_nearest)
                    final_label = label if current_min <= min_distance else final_label
                    min_distance = min(current_min, min_distance)

            return final_label


    def predict(self, predict_data) -> List[Union[str, int]]:
        if not self._fitted:
            raise RuntimeError("The classifier must be fitted before predicting")

        # process the data
        predict_data = self._preprocess(predict_data)

        return [self._predict(image) for image in predict_data]