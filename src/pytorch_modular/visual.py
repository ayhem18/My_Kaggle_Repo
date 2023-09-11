"""
This script contains functionalities needed for visualization and debugging purposes
"""

import matplotlib.pyplot as plt
import numpy as np

from _collections_abc import Sequence
from typing import Union, List

import torch


def plot_images(images: Union[np.ndarray, torch.Tensor, Sequence[np.ndarray]],
                captions=List[str],
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
        if i < len(captions):
            plt.title(captions[i])
        plt.axis("off")
    fig.suptitle(title)
    plt.show()
