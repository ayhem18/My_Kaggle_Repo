"""
This script contains functionalities to implement an unconventional parametric classifier
"""

import random
import numpy.random
import os

import numpy as np

from typing import Tuple

script_dir = os.path.dirname(os.path.realpath(__file__))


# a function to load the data into numpy arrays
def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from torchvision import datasets, transforms as tr
    import torch

    if not os.path.isdir(os.path.join(script_dir, 'data')):
        raise ValueError(f"Please create a directory 'data' in the script's parent directory")

    data_folder = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_folder, exist_ok=True)

    # apply toTensor transform to each of the images
    basic_transform = tr.Compose([tr.ToTensor()])
    # use Pytorch to load the dataset
    cifar_train = datasets.CIFAR10(root=os.path.join(data_folder, 'train'), train=True, download=True,
                                   transform=basic_transform)
    cifar_test = datasets.CIFAR10(root=os.path.join(data_folder, 'test'), train=False, download=False,
                                  transform=basic_transform)

    # build a loader for each split
    from torch.utils.data import DataLoader
    # convert the dataset to a Dataloader for easier manipulation
    train_loader = DataLoader(cifar_train, batch_size=1000, shuffle=False)
    test_loader = DataLoader(cifar_test, batch_size=1000, shuffle=False)

    train_tensor = torch.stack([data for data, _ in train_loader])
    test_tensor = torch.stack([data for data, _ in test_loader])

    train_np = train_tensor.permute((0, 1, 3, 4, 2)).reshape(shape=(len(cifar_train), -1)).numpy()
    test_np = test_tensor.permute((0, 1, 3, 4, 2)).reshape(shape=(len(cifar_test), -1)).numpy()

    train_labels = torch.stack([labels for _, labels in train_loader]).reshape((-1,)).numpy()
    test_labels = torch.stack([labels for _, labels in test_loader]).reshape((-1,)).numpy()

    # the normalization of the image is necessary to avoid as many numerical issues as possible
    # since softmax is really prone to the overflow problem
    return train_np / 255, test_np / 255, train_labels, test_labels


# the next 2 functions are not technically ones of the classifier's functionalities
# and thus will be written outside the class definition
def softmax(array: np.ndarray):
    if len(array.shape) != 1:
        raise ValueError(f"The input is expected to be 1 dimensional. Found: {len(array.shape)} dimensions")

    sum_exp = np.sum(np.exp(array))
    return np.exp(array) / sum_exp


def cross_entropy_loss(probs: np.ndarray, y_true: np.ndarray) -> float:
    y_true = y_true.squeeze()
    # as this condition is satisfied, proceed with
    selection = probs[np.arange(len(probs)), y_true]
    # calculate the cross entropy loss
    loss = -np.sum(np.log(selection))
    return loss


# write a function to compute the accuracy of some prediction against given labels
def compute_accuracy(predictions: np.ndarray, y_true: np.ndarray) -> float:
    return np.mean((predictions == y_true))


class WeirdClassifier:
    def __init__(self,
                 input_features: int,
                 num_classes: int,
                 choices: int = 10 ** 5,
                 seed: int = 69):
        # set the seed for reproducibility
        numpy.random.seed(seed)
        random.seed(seed)
        # number of input units
        self.in_features = input_features
        # number of output units: depends on the number of classes
        self.out_features = num_classes if num_classes > 2 else num_classes
        # the weights matrix: of shape
        self.w = np.random.rand(self.in_features, self.out_features)
        self.choices = choices
        self.fit = False

    def _compute_probs(self, x: np.ndarray) -> np.ndarray:
        # 2 conditions must be satisfied: all values must be in the range [0, 1]
        # the input must be a 2-dimensional matrix of shape : batch_size, self.in_features
        if ((x < 0) & (x > 1)).any():
            raise ValueError("The input is expected to have all values within the range [0, 1]")

        if len(x.shape) != 2 or x.shape[1] != self.in_features:
            raise ValueError(f'The function expects a matrix with dimensions: {(None, self.in_features)}'
                             f'\nFound: {x.shape}')

        logits = x @ self.w
        # apply the softmax operation to each row
        probs = np.apply_along_axis(softmax, axis=1, arr=logits)
        return probs

    def train(self,
              x: np.ndarray,
              y: np.ndarray,
              seed: int = 69):

        min_loss = float('inf')
        best_weight = None
        for _ in range(self.choices):
            self.w = np.random.rand(self.in_features, self.out_features)
            # calculate the predictions and loss
            predictions = self._compute_probs(x)
            loss = cross_entropy_loss(predictions, y)
            # update weight
            best_weight = self.w if best_weight is None or loss < min_loss else best_weight
            # update loss
            min_loss = min(loss, min_loss)

        self.fit = True
        # set the weight
        self.w = best_weight

    def predict(self,
                x: np.ndarray,
                requires_train: bool = True):

        if requires_train and not self.fit:
            raise RuntimeError(f"The model must be trained before performing predictions")

        probs = self._compute_probs(x)
        preds = np.argmax(probs, axis=1)
        return preds


if __name__ == '__main__':
    train_x, test_x, train_y, test_y = load_data()
    classifier = WeirdClassifier(input_features=3072, num_classes=10)
    classifier.train(train_x, train_y)
    y_pred = classifier.predict(test_x)
    # evaluate the model
    accuracy = compute_accuracy(y_pred, test_y)
    print(f"the model achieves an accuracy of {accuracy}")
