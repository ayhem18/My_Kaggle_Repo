"""
This script contains functionalities to wrap images using affine transformations
"""
import os

import numpy as np
import cv2 as cv

from typing import Union, List, Tuple
from _collections_abc import Sequence
from collections import defaultdict
from scipy.spatial.distance import euclidean, cityblock
from math import ceil
from pathlib import Path
from src.CV_from_scratch.geometric_transformations import geometric_transformer as gt

BI_LINEAR = 'bi-linear'
NEAREST_NEIGHBORS = 'nearest'

num = Union[int, float]


# let's define a function that will the extremes of one domain to another
def new_domain_extremes(extremes: List[Union[Tuple[int], num]], matrix: np.array) -> List[Tuple[int, float]]:
    # first let's start with validating the input
    # allow converting a single number into a tuple
    extremes = [(t, t) if isinstance(t, (int, float)) else sorted(t) for t in extremes]
    # make sure the matrix is 2 * 2
    if len(matrix.shape) != 2:
        raise ValueError(f'The matrix must be 2-dimensional')

    # make sure the lengths match:
    if len(extremes) != len(matrix[1]):
        raise ValueError("the function expects the extremes to match the number of rows in the matrix")

    # we need to define a maximum and minimum for each new dimension
    def new_extreme_1_dim(coefs: Sequence[int, float]) -> Tuple[int, int]:
        # coefs represents the matrix coefficients
        # the 'extremes' variable are the same as the 'extremes' passed to the parent function
        min_new_extreme = sum([min(c * t[0], c * t[1]) for c, t in zip(coefs, extremes)])
        max_new_extreme = sum([max(c * t[0], c * t[1]) for c, t in zip(coefs, extremes)])
        return min_new_extreme, max_new_extreme

    res = [new_extreme_1_dim(row_coefs) for row_coefs in matrix]
    return res


class Interpolator:
    @classmethod
    def _list_candidates(cls,
                         reverse_y: num,
                         reverse_x: num,
                         image: np.array) -> List[Tuple[int, int]]:

        # each pair of coordinates has 4 theoretical candidates resulting from applying either the floor
        # or ceil operators to either coordinate
        higher_y, lower_y, higher_x, lower_x = (int(ceil(reverse_y)), int(reverse_y),
                                                int(ceil(reverse_x)), int(reverse_x))

        candidates = [(higher_y, higher_x),
                      (higher_y, lower_x),
                      (lower_y, higher_x),
                      (lower_y, lower_x)]

        # extract image dimensions
        h, w = image.shape[:2]

        # filter candidates lying out of the image boundary
        return list(set([c for c in candidates if 0 <= c[0] < h and 0 <= c[1] < w]))

    def __init__(self,
                 interpolation: Union[str, int],
                 original_image: np.array = None) -> None:

        if isinstance(interpolation, str) and interpolation not in [BI_LINEAR, NEAREST_NEIGHBORS]:
            raise ValueError(f"a string interpolation argument should be either: {NEAREST_NEIGHBORS}"
                             f"of {BI_LINEAR}.\nFound: {interpolation}")

        if isinstance(interpolation, int):
            self.interpolation = NEAREST_NEIGHBORS if interpolation == 0 else BI_LINEAR
        else:
            self.interpolation = interpolation

        # set an image as a field
        self.image = original_image

        # initialize a filed to map each interpolation constant to the actual function
        self.interpolation_map = defaultdict(lambda: self._interpolate_nearest_neighbor)
        self.interpolation_map[NEAREST_NEIGHBORS] = self._interpolate_nearest_neighbor
        self.interpolation_map[BI_LINEAR] = self._interpolate_bi_linear

    def _interpolate_nearest_neighbor(self,
                                      reverse_y: num,
                                      reverse_x: num,
                                      image: np.array = None
                                      ) -> Union[int, np.ndarray]:
        # find the possible candidates
        candidates = self._list_candidates(reverse_y, reverse_x, image)

        # let's account for the possibility that are no valid candidates
        if len(candidates) == 0:
            return 0

        # out of the suggested candidates, select the pixel with the closest coordinate: manhattan distance
        pixel_coordinates = min(candidates, key=lambda c: cityblock((reverse_y, reverse_x), c))

        return image[pixel_coordinates[0], pixel_coordinates[1], :]

    def _interpolate_bi_linear(self,
                               reverse_y: num,
                               reverse_x: num,
                               image: np.array = None,
                               ) -> Union[int, np.ndarray]:
        # bi-linear interpolation can be carried out as follows:
        # 1. choose the coordinates (common stage)
        candidates = self._list_candidates(reverse_y, reverse_x, image)
        if len(candidates) == 0:
            return 0
        # 2. measure the Euclidean distance to each candidate
        distances = [euclidean((reverse_y, reverse_y), c) for c in candidates]
        # average by the total
        total_distance = sum(distances)
        # transpose as now distances of shape = (1, len(c))
        distances = np.array([[d / total_distance for d in distances]]) \
            if total_distance > 0 else np.array([[1]], dtype=np.float32)

        # extract the pixel values of each candidate across all channels:
        # image_array should have the shape (len(c), 1, num_channels)
        image_array = np.asarray([image[c[0], c[1], :] for c in candidates])

        # reshape to (1, len(c), num_channels) for the matrix multiplication to be mathematically correct
        # image_array = np.reshape(image_array, newshape=(1, len(candidates), image.shape[-1]))

        result = distances @ image_array

        return result

    def interpolate(self,
                    reverse_y: num,
                    reverse_x: num,
                    original_image: np.array,
                    interpolation: str = None
                    ) -> Union[int, np.ndarray]:
        image = self.image if original_image is None else original_image

        if image is None:
            raise TypeError(f'The interpolator class requires either to pass an image or set the original_image field'
                            f'Both are {None} objects')

        # this method will simply call the corresponding inner method
        return (self.interpolation_map[self.interpolation if interpolation is None else interpolation]
                (reverse_y, reverse_x, image=original_image))


class Wrapper:
    @classmethod
    def coordinate_float_to_int(cls, coordinate: Union[int, float]) -> int:
        # since blindly applying the ceil operator might lead to unexpected behavior
        r_int = int(round(coordinate))
        coordinate = r_int if np.abs(r_int - coordinate) <= 10 ** -2 else coordinate
        # the main idea is to stretch the value in its direction
        return int(np.sign(coordinate) * ceil(np.abs(coordinate)))

    @classmethod
    def standard_image(cls, image: np.array) -> np.array:
        """
        make sure the numpy array passed represents image data: dimensions-wise.
        add an extra dimension if the image is gray scale
        Args:
            image: a numpy array that represents an actual image

        Returns: either the image with an extra dimension (if it is grayscale) or the original image
        """
        if not (len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[-1] == 3)):
            raise ValueError("The input is expected to represent an image: The dimensions are either 1 (gray scale)"
                             f" or 3 with 3 color channels\n Found {len(image.shape)} dimensions"
                             + (f'with {str(image.shape[-1])}color channels' if len(image.shape) == 3 else ''))

        return np.expand_dims(image, axis=-1) if len(image.shape) == 2 else image

    def __init__(self,
                 original_image: np.array,
                 interpolation=None) -> None:
        self.original_image = self.standard_image(original_image)
        self.interpolator = Interpolator(interpolation)

    def _new_extremes(self,
                      transformation: np.array,
                      image: np.array,
                      ) -> Tuple[int, int, int, int, int]:
        if image is not None:
            self.image = image

        image = self.image

        # extract the image dimensions
        h, w, c = image.shape

        # find the dimensions of the new image
        # keep in mind that the transformation is applied to [x, y, 1] and not [y, x, 1]
        new_extremes = new_domain_extremes([(0, w - 1), (0, h - 1), 1], transformation)
        # unzip the new dimensions
        new_min_y, new_max_y = new_extremes[1]
        new_min_x, new_max_x = new_extremes[0]

        # make sure to convert them to integer values
        new_min_y = self.coordinate_float_to_int(new_min_y)
        new_max_y = self.coordinate_float_to_int(new_max_y)

        new_min_x = self.coordinate_float_to_int(new_min_x)
        new_max_x = self.coordinate_float_to_int(new_max_x)

        return new_min_y, new_max_y, new_min_x, new_max_x, c

    def wrap(self,
             transformation: np.array,
             inverse_transformation: np.array,
             image: np.array = None,
             ) -> np.array:

        image = self.original_image if image is None else image

        # first extract the different values needed
        new_min_y, new_max_y, new_min_x, new_max_x, c = self._new_extremes(transformation, image)

        new_y, new_x = new_max_y - new_min_y + 1, new_max_x - new_min_x + 1
        new_shape = (new_y, new_x, c)
        new_image = np.zeros(shape=new_shape, dtype=np.uint8)

        # the above 'for loop' implementation is too slow, let's try implementing some matrix multiplication
        theo_coordinates_matrix = np.array(
            [[x + new_min_x, y + new_min_y, 1] for y in range(new_y) for x in range(new_x)]
        ).transpose()
        # now our coordinates are as follows:

        # y_theo = 0 for new_x consecutive x values
        # y_theo = 1 for new_x consecutive x values
        # y_theo = 2 ...

        # theo_coordinates should be of the shape: (3, new_x * new_y)
        assert theo_coordinates_matrix.shape == (3, new_x * new_y)

        reverse_coordinates_matrix = inverse_transformation @ theo_coordinates_matrix
        # reverse_coordinates_matrix is of shape (3, new_x * new_y)
        assert reverse_coordinates_matrix.shape == (3, new_x * new_y)

        for y in range(new_y):
            for x in range(new_x):
                col = y * new_x + x
                reverse_x, reverse_y, _ = reverse_coordinates_matrix[:, col].squeeze()
                # set the pixel value
                new_image[y, x, :] = self.interpolator.interpolate(reverse_y, reverse_x, image)

        return new_image


# time to see our wrapping functionality in action
def main():
    image_path = os.path.join(Path(os.getcwd()).parent, 'cat_image.jpg')
    img = cv.imread(image_path)#, cv.IMREAD_GRAYSCALE)
    cv.imshow('image', img)
    cv.waitKey(0)

    img_np = np.asarray(img)
    print(img_np.shape)

    # first let's define the transformation matrix and its inverse
    geo_transformer = gt.GeometricTransformer2D(image=img_np)
    rot_mat = geo_transformer.get_transformation_matrix('rotation', 40)
    inv_rot_mat = geo_transformer.get_inverse_transformation_matrix(rot_mat, transformation='rotation')

    print(rot_mat)
    print(inv_rot_mat)

    w = Wrapper(original_image=img_np, interpolation=BI_LINEAR)

    rotated_image = w.wrap(rot_mat, inv_rot_mat)

    cv.imshow('rotated_image', rotated_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
