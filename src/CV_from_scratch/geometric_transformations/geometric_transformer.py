"""
This script contains functionalities to apply geometric transformations on a given image.
The process is also referred to as wrapping.

I learnt the general design the wrapping process from the following short video series:
https://youtu.be/l_qjO4cM74o?si=Zm__L7GtiCrmbTk8
"""

from pathlib import Path
from typing import Union, Tuple

import cv2 as cv
import numpy as np

num = Union[float, int]


class GeometricTransformer2D:
    __three_three = (3, 3)
    __two_three = (2, 3)
    __two_two = (2, 2)

    ROTATION = 'rotation'
    SKEW = 'skew'
    SCALE = 'scale'
    TRANSLATION = 'translation'

    @classmethod
    def rotation_2d(cls, angle: num, degrees: bool = True) -> np.ndarray:
        # first convert the angle to radians if needed
        # TODO: Verify if applying the modulo operator is needed: to keep angle within the [0, 360] range

        angle = np.deg2rad(angle) if degrees else angle
        cos, sin = np.cos(angle), np.sin(degrees)
        # the class method will return a 3 * 3 matrix:
        return np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])

    @classmethod
    def translation_2d(cls,
                       tx: num = None,
                       ty: num = None) \
            -> np.ndarray:

        if tx is None and ty is None:
            raise TypeError(f"At least one of the translation parameters should be passed\nFound:"
                            f" tx: {tx}\nty: {ty}")

        # passing only one of these parameters, meaning the same parameter is applied on both dimensions
        tx = tx if tx is not None else ty
        ty = ty if ty is not None else tx

        return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)

    @classmethod
    def scale_2d(cls,
                 sx: num = None,
                 sy: num = None) \
            -> np.ndarray:
        if sx is None and sy is None:
            raise TypeError(f"At least one of the scaling parameters should be passed\nFound:"
                            f" sx: {sx}\nsy: {sy}")

        if sy <= 0 or sx <= 0:
            raise ValueError(f"The scaling parameters are expected to be strictly positive\nFound:"
                             f"sy: {sy}, sx: {sx}")

        # passing only one of these parameters implies applying the same parameter on both dimensions
        sx = sx if sx is not None else sy
        sy = sy if sy is not None else sx

        return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float32)

    @classmethod
    def skew_2d(cls,
                mx: num = None,
                my: num = None) \
            -> np.ndarray:
        if mx is None and my is None:
            raise TypeError(f"At least one of the skew parameters should be passed\nFound:"
                            f" tx: {mx}\nty: {my}")
        # the default behavior of this function is slightly different from 'scale' or 'translate'
        # a parameters that wasn't passed explicitly will be set to '0'

        mx = mx if mx is None else 0
        my = my if my is None else 0

        return np.array([[1, my, 0], [mx, 1, 0], [0, 0, 1]], dtype=np.float32)

    def __init__(self, image: Union[str, Path, np.ndarray], matrix_shape: Tuple[int, int] = None):
        """
        The constructor saves:
        image_path: [str, Path], the path of the image if given
        image: np.nd of the image
        Args:
            image: the image either np.nd of path to an actual image
        """

        self.image_path = image if isinstance(image, (Path, str)) else None
        self.image = image if isinstance(image, np.ndarray) else np.array(cv.imread(image))

        self.matrix_shape = matrix_shape if matrix_shape is not None else self.__three_three
        self.transform_matrix = None

        self.transform_map = {self.ROTATION: self.rotation_2d,
                              self.SKEW: self.skew_2d,
                              self.TRANSLATION: self.translation_2d,
                              self.SCALE: self.scale_2d}

    def set_shape(self, shape: Tuple[int, int]) -> Tuple[int, int]:
        shape = self.matrix_shape if shape is None else shape
        s = [self.__three_three, self.__two_three, self.__two_two]
        if shape not in s:
            raise ValueError(f'The shape is expected to be one of the following shapes {s}\nFound {shape}')
        return shape

    def get_rotation_matrix(self,
                            angle: Union[float, int],
                            degrees: bool = True,
                            shape: Tuple[int, int] = None) -> np.array:
        shape = self.set_shape(shape)
        rows, cols = shape
        matrix = self.rotation_2d(angle, degrees)[:rows, :cols]

        if self.transform_matrix is None:
            self.transform_matrix = matrix
        return matrix

    def get_translation_matrix(self,
                               tx: num = None,
                               ty: num = None,
                               shape: Tuple[int, int] = None) -> np.array:

        shape = self.set_shape(shape)
        rows, cols = shape

        matrix = self.translation_2d(tx, ty)[:rows, :cols]

        if self.transform_matrix is None:
            self.transform_matrix = matrix
        return matrix

    def get_scale_matrix(self,
                         sx: num = None,
                         sy: num = None,
                         shape: Tuple[int, int] = None) -> np.array:

        shape = self.set_shape(shape)
        rows, cols = shape

        matrix = self.scale_2d(sx, sy)[:rows, :cols]

        if self.transform_matrix is None:
            self.transform_matrix = matrix
        return matrix

    def get_skew_matrix(self,
                        mx: num = None,
                        my: num = None,
                        shape: Tuple[int, int] = None) -> np.array:

        shape = self.set_shape(shape)
        rows, cols = shape

        matrix = self.skew_2d(mx, my)[:rows, :cols]

        if self.transform_matrix is None:
            self.transform_matrix = matrix
        return matrix

    def get_transformation_matrix(self, transformation: str, *args) -> np.array:
        if transformation.lower() not in list(self.transform_map.keys()):
            raise ValueError("The operation is expected to be one of the following:\n"
                             f"{list(self.transform_map.keys())}\nFound: {transformation.lower()}")

        return self.transform_map[transformation.lower()](*args)

    def get_inverse_transformation_matrix(self, matrix: np.ndarray, operation: str = None) -> np.ndarray:
        if matrix.shape == self.__two_three:
            matrix = np.append(matrix, np.asarray([0, 0, 1]), axis=0)

        mat = matrix.copy()

        if operation in [self.ROTATION, self.SKEW]:
            mat[0][1] = -matrix[0][1]
            mat[1][0] = -matrix[1][0]

        elif operation == self.SCALE:
            mat[0][0] = 1 / matrix[0][0]
            mat[1][1] = 1 / matrix[1][0]

        elif operation == self.TRANSLATION:
            mat[0][-1] = -matrix[0][-1]
            mat[1][-1] = -matrix[1][-1]

        else:
            # in this case the matrix is not an elementary
            # affine transformation
            mat = np.linalg.inv(matrix)

        # make sure the calculation are correct
        assert np.allclose(np.identity(mat.shape[0]) - mat @ matrix, np.zeros(mat.shape)), \
            "MAKE SURE THE INVERSE IS COMPUTED CORRECTLY"

        return mat
