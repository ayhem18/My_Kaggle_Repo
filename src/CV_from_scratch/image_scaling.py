"""
This script contains different algorithms written from scratch to down/up scales images
"""
import math
import os
import numpy as np
from scipy.spatial.distance import cityblock

from typing import Union
from PIL import Image


def scale_nearest_neighbors(image: np.array,
                            new_h: int = None,
                            new_w: int = None,
                            ratio: Union[float, int] = 2
                            ) -> np.array:
    # make sure either both new_w and new_h are passed
    # or none of them
    if (new_h is None) != (new_w is None):
        raise TypeError(f"Either both new dimensions are passed or none. \nFound "
                        f"new_h: {new_h}, new_w: {new_w}")

    # the ratio must be positive
    if ratio <= 0:
        raise ValueError(f"The function expects a positive ratio\nFound: {ratio}")

    if len(image.shape) not in [2, 3] or (len(image.shape) == 3 and image.shape[-1] not in [3, 1]):
        raise ValueError(f"This function was written for images with 1 or 3 channels")

    # extract the shape of the image
    if len(image.shape) == 2:
        h, w = image.shape
    else:
        return np.array([scale_nearest_neighbors(image[:, :, i]) for i in range(image.shape[-1])])

    if new_h is None:
        new_h = int(h / ratio)
        new_w = int(w / ratio)

    else:
        ratio_h = new_h / h
        ratio_w = new_w / w

    new_image = np.zeros((new_h, new_w))


def scale_nearest_neighbors(image: np.array,
                            new_h: int = None,
                            new_w: int = None,
                            ratio: Union[float, int] = 2) -> np.array:
    if len(image.shape) == 2:
        # add a dimension to the image
        image = np.expand_dims(image, axis=-1)

    # extract the different dimensions
    # h: height, w:width, c: channels
    h, w, c = image.shape

    if new_h is None and new_w is None:
        new_h, new_w = int(math.ceil((ratio * h))), int(math.ceil((ratio * w)))
        ratio_h, ratio_w = ratio, ratio
    else:
        ratio_h = new_h / h
        ratio_w = new_w / w

    new_image = np.zeros((new_h, new_w, c), dtype=int)

    for cn in range(c):
        for y in range(new_h):
            for x in range(new_w):
                new_image[y][x][cn] = image[int(y / ratio_h)][int(x / ratio_w)][cn]

    # make sure to squeeze the image again (in case the original image was gray scale)

    return np.squeeze(new_image)


if __name__ == '__main__':
    image_path = os.path.join(os.getcwd(), 'cat_image.jpg')
    image = Image.open(image_path)
    image.show(title='original.png')

    im_np = np.asarray(image)
    print(im_np.shape)
    y = scale_nearest_neighbors(im_np, im_np.shape[0] // 2, im_np.shape[1] // 2).astype(np.uint8)

    Image.fromarray(y).show(title='scaled_me.png')
