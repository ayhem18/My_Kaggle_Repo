"""
This script contains functionalities needed for visualization and debugging purposes
"""
import random

import torch
import os

import matplotlib.pyplot as plt
import numpy as np
import cv2

from _collections_abc import Sequence
from typing import Union, List, Optional
from pathlib import Path
from PIL import Image

random.seed(69)

POSSIBLE_BACKENDS = ['PIL', 'opencv']


def _image_to_np(image_path: Union[str, Path], backend: str = 'PIL') -> np.ndarray:
    if backend not in POSSIBLE_BACKENDS:
        raise ValueError(f"the backed is expected to be one of the following: {POSSIBLE_BACKENDS}\nFound: {backend}")

    if backend == 'PIL':
        return np.asarray(Image.open(image_path))

    return np.asarray(cv2.imread(image_path))


def plot_images(images: Union[np.ndarray, torch.Tensor, Sequence[np.ndarray]],
                captions:Optional[List] = None,
                rows=2,
                columns=5,
                title="", **kwargs):
    """
    Plots images with captions

    :param images: list of images to plot
    :param captions: captions of images:
    :param rows: number of rows in figure
    :param columns: number of columns:
    :param title: super title of figure
    """
    fig = plt.figure(figsize=(6, 3))
    for i, img in enumerate(images):
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(img, **kwargs)

        if captions is not None:
            if i < len(captions):
                plt.title(captions[i])
        
        plt.axis("off")
    fig.suptitle(title)
    plt.show()


def explore_train_data(train_directory: Union[Path, str],
                       image_per_class: int = 5,
                       seed: int = 69,
                       backend: str = 'PIL') -> None:
    # set the seed
    random.seed(seed)

    def all_inner_files_directories(path):
        return all([
            os.path.isdir(os.path.join(path, d)) for d in os.listdir(path)
        ])

    if not all_inner_files_directories(train_directory):
        raise ValueError("The input directory is expected to be a training directory with classes")

    for cls in os.listdir(train_directory):
        cls_directory = os.path.join(train_directory, cls)
        # get a random sample of the images in the current class
        image_samples = random.sample(os.listdir(cls_directory), k=image_per_class)
        # convert the image's name to an absolute path
        image_samples = [os.path.join(cls_directory, i) for i in image_samples]
        # convert the images to numpy arrays
        images_as_nps = [_image_to_np(img) for img in image_samples]
        plot_images(images=images_as_nps,
                    title=cls)
