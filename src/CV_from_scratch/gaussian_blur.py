# the main idea here is to try to write the gaussian blur from scratch in python

import numpy as np
from typing import Union, Tuple


def gaussian_kernel(kernel_size: Union[int, Tuple[int, int]],
                    sigma: float,
                    y_0: int = None,
                    x_0: int = None) -> np.array:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if kernel_size[0] % 2 == 0 or kernel_size[0] % 2 == 0:
        raise ValueError("Both kernel dimensions are expected to be odd.\n"
                         f"Found: width = {kernel_size[1]}, height = {kernel_size[1]}")

    # the default values for x_0 and y_0 is the middle of the kernel
    y_0 = kernel_size[0] // 2 if y_0 is None else y_0
    x_0 = kernel_size[1] // 2 if x_0 is None else x_0

    h, w = kernel_size
    # create the numpy array with the exponents
    kernel = [((y - y_0) ** 2 + (x - x_0) ** 2) for x in range(w) for y in range(h)]
    # convert to numpy array
    kernel = np.asarray(kernel)

    kernel = -kernel / (2 * sigma ** 2)

    # divide by the constant operator
    kernel = kernel / (2 * np.pi * sigma ** 2)

    return kernel


def convolution_op(image_np: np.array, kernel: np.array) -> np.array:
    h, w = image_np.shape
    k_h, k_w = kernel.shape

    # calculate the dimensions of the output


# ny, nx = y / ratio_h, x / ratio_w
#
# higher_ny, lower_ny, higher_nx, lower_nx = int_dim(ny, ratio=1), int(ny), int_dim(nx, ratio=1), int(nx)
#
# neighbors = [(higher_ny, higher_nx),
#              (higher_ny, lower_nx),
#              (lower_ny, higher_nx),
#              (lower_ny, lower_ny)]
# # filter by the boundaries
# neighbors = [n for n in neighbors if 0 < n[0] < new_h and 0 < n[1] < new_w]
# # calculate the Euclidean distance to each of the valid neighbors
# distances = [euclidean((ny, nx), n) for n in neighbors]
# total_distance = sum(distances)
# distances = np.array([d / total_distance for d in distances]).squeeze().transpose()
#
# new_image[padding_y + y, padding_x + x, :] = (np.asarray([image[n[0], n[1], :] for n in neighbors])
#                                               @ distances)
