"""
This script contains functionalities to perform 2 * 2 transformations on images
"""
import math
import os

import numpy as np
import cv2 as cv

from typing import Union, Tuple
from pathlib import Path


def coordinate_float_to_int(coordinate: Union[int, float]) -> int:
    # since blindly applying the ceil operator might lead to unexpected behavior
    r_int = int(round(coordinate))
    coordinate = r_int if np.abs(r_int - coordinate) <= 10 ** -2 else coordinate
    # the main idea is to stretch the value in its direction
    return int(np.sign(coordinate) * int(math.ceil(np.abs(coordinate))))


def new_image_domain(min_y: int, max_y: int, min_x, max_x, matrix: np.array) -> Tuple[int, int, int, int]:
    """
    Since a 2 * 2 transformation operates on the domain of the image (its coordinates), the extremes of the
    new domain should be computed for the image to be displayed
    Args:
        min_x: minimum coordinate 'x' in the original domain
        max_x: maximum coordinate 'x' in the original domain
        min_y: minimum coordinate 'y' in the original domain
        max_y: maximum coordinate 'y' in the original domain
        matrix: the 2 * 2 transformation matrix

    Returns: The extremes of the new domain
    """
    # first let's check if the matrix is 2 * 2

    if matrix.shape != (2, 2):
        raise ValueError("The program expects a 2 * 2 transformation matrix\nFound matrix of "
                         f"shape: {matrix.shape}")

    # what makes this problem simple is that 'x' and 'y' are independent:
    # let's define an inner function to solve the problem with only one new coordinate

    def new_domain_1_dim(coef1: float, coef2: float) -> Tuple[int, int]:
        # the max value of the given dimension can be found with a simple formula
        max_value = coef1 * (min_x if coef1 <= 0 else max_x) + coef2 * (min_y if coef2 <= 0 else max_y)
        min_value = coef1 * (min_x if coef1 > 0 else max_x) + coef2 * (min_y if coef2 > 0 else max_y)

        return coordinate_float_to_int(max_value), coordinate_float_to_int(min_value)

    c11, c12 = matrix[0]
    c21, c22 = matrix[1]

    new_max_x, new_min_x = new_domain_1_dim(c11, c12)
    new_max_y, new_min_y = new_domain_1_dim(c21, c22)

    return new_min_y, new_max_y, new_min_x, new_max_x


def rotate(image: np.array, angle: Union[float, int], is_degree: bool = True) -> np.array:
    angle = np.deg2rad(angle) if is_degree else angle
    # compute the cos and sin values
    cos = np.cos(angle)
    sin = np.sin(angle)

    rotation_matrix = np.array([[cos, -sin], [sin, cos]])

    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)

    rows, cols, channels = image.shape

    # let's determine the new domain
    min_y, max_y, min_x, max_x = new_image_domain(min_y=0,
                                                  max_y=rows - 1,
                                                  min_x=0,
                                                  max_x=cols - 1,
                                                  matrix=rotation_matrix)

    # the new matrix should have the dimensions: max_y - min_y + 1, max_x - min_x + 1
    # THE INTEGER TYPE IS CRUCIAL FOR THE ARRAY TO BE RECOGNIZED AS AN IMAGE
    new_image = np.zeros(shape=(max_y - min_y + 1, max_x - min_x + 1, channels), dtype=np.uint8)

    # iterate through the original image and map each pixel to its destination pixel
    for y in range(rows):
        for x in range(cols):
            # flip the coordinates to match the theoretical ones
            y_theo = rows - 1 - y

            res_vector = rotation_matrix @ np.array([[x], [y_theo]])
            nx, ny = np.squeeze(res_vector)
            nx, ny = coordinate_float_to_int(nx), coordinate_float_to_int(ny)

            nx, ny = nx - min_x, ny - min_y
            # flip ny once again
            ny = max_y - ny

            # subtract min_y, min_x from each pair of coordinates: to keep the new coordinates within
            # the array boundaries
            new_image[ny, nx, :] = image[y, x, :]

    if image.shape[-1] == 1:
        new_image = np.squeeze(new_image)
    return new_image


def main():
    image_path = os.path.join(Path(os.getcwd()).parent, 'cat_image.jpg')
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    cv.imshow('image', img)
    cv.waitKey(0)

    im_np = np.asarray(img)
    print(im_np.shape)
    y = rotate(im_np, angle=30)
    cv.imshow('rotated_image', y)
    cv.waitKey(0)
    cv.destroyAllWindows()


# let's see the code in action
if __name__ == '__main__':
    # img = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    # print(img)
    # new_img = rotate(img, angle=30)
    # print(new_img)
    main()
