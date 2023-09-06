"""
This script contains different algorithms written from scratch to down/up scales images
"""
import math
import os
import numpy as np
import cv2 as cv

from typing import Union

NEAREST_NEIGHBORS = 'nearest'
BI_LINEAR = 'bi_linear'


def int_dim(dimension: Union[float, int], ratio: Union[float, int]) -> int:
    """
    This function is helped function to avoid writing the unpleasant expression below multiple times
    """
    return int(math.ceil((ratio * dimension)))


def scale_nearest_neighbors(image: np.array,
                            new_h: int = None,
                            new_w: int = None,
                            ratio: Union[float, int] = 2,
                            keep_ratio: bool = True,
                            ) -> np.array:
    """
    This function scales a given image using a simplified version of the nearest neighbors interpolation
    Args:
        image: the given image
        new_h: the height of the output
        new_w: the width of the output
        ratio: the ratio: used when the dimensions are not passed
        keep_ratio: determines whether the output image should have the same dimensions as the input while
        the dimensions with larger ratio is padded with black pixels

    Returns:
    an image scaled according to new_h / h or new_w / w or simply ratio
    """
    # let's make sure the input is valid
    if not (len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[-1] == 3)):
        raise ValueError("The input is expected to represent an image: The dimensions are either 1 (gray scale)"
                         f" or 3 with 3 color channels\n Found {len(image.shape)} dimensions "
                         f"{(('with ' + str(image.shape[-1]) + 'color channels') if len(image.shape) == 3 else '')}")

    # to avoid duplicating code, we will expand the dimension of gray images
    if len(image.shape) == 2:
        # add a dimension to the image
        image = np.expand_dims(image, axis=-1)

    # extract the different dimensions
    # h: height, w:width, c: channels
    h, w, c = image.shape

    if new_h is None and new_w is None:
        new_h, new_w = int_dim(h, ratio), int_dim(w, ratio)
        ratio_h, ratio_w = ratio, ratio
    else:
        if new_h is None or new_w is None:
            raise TypeError("The function expects either both or none of the new dimensions to be passed")
        ratio_h = new_h / h
        ratio_w = new_w / w

    # the output image will differ depending on the value of the 'keep_ratio' argument
    if keep_ratio:
        # first find the minimal ratio
        min_ratio = min(ratio_w, ratio_h)

        # find the new dimension of the image (without padding)
        effective_h, effective_w = int_dim(min_ratio, h), int_dim(min_ratio, w)

        # ratio_h and ratio_w are shared between both cases, so they should be set to min_ratio in this case
        ratio_h = min_ratio
        ratio_w = min_ratio

        # since the non-padded part is expected to be smalled than the output dimensions, we need to specify
        # the number of budding pixels for both the 'x' and 'y' axis
        padding_y = abs(effective_h - new_h) // 2
        padding_x = abs(effective_w - new_w) // 2

    else:
        effective_h, effective_w = new_h, new_w
        padding_y, padding_x = 0, 0

    new_image = np.zeros(shape=(new_h, new_w, c), dtype=np.uint8)

    for y in range(effective_h):
        for x in range(effective_w):
            # the idea here is to assign the target pixel, the pixel in the original image
            # found by the inverse of the scaling transformation: scaling by the inverse of the ratio
            new_image[padding_y + y, padding_x + x, :] = image[int(y / ratio_h), int(x / ratio_w), :]

    # make sure to squeeze the image again (in case the original image was gray scale)
    return np.squeeze(new_image)


if __name__ == '__main__':
    image_path = os.path.join(os.getcwd(), 'cat_image.jpg')
    # uncomment to see the algorithm output with grayscale images as well
    img = cv.imread(image_path) #,cv.IMREAD_GRAYSCALE)
    cv.imshow('image', img)
    cv.waitKey(0)

    im_np = np.asarray(img)
    h, w = im_np.shape[:2]
    print(im_np.shape)

    # y = scale_nearest_neighbors(im_np, new_h=int(h * 1.2), new_w=int(w * 1.6), keep_ratio=True)
    # other possible calls:
    # y = scale_nearest_neighbors(im_np, new_h=int(h * 1.6), new_w=int(w * 1.2), keep_ratio=True)
    # y = scale_nearest_neighbors(im_np, new_h=int(h * 0.4), new_w=int(w * 0.8), keep_ratio=True)
    y = scale_nearest_neighbors(im_np, new_h=int(h * 0.5), new_w=int(w * 0.2), keep_ratio=True)

    # y = scale_nearest_neighbors(im_np, ratio=1.8, keep_ratio=True)
    print(y.shape)
    cv.imshow('rotated_image', y)
    cv.waitKey(0)
    cv.destroyAllWindows()
