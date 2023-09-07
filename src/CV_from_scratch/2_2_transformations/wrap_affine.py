"""
This script is written to basically explore wrapping and affine transformations
"""

from __future__ import print_function

import os

import cv2 as cv
import numpy as np


def main(image_path: str):
    src_image = cv.imread(image_path)

    srcTri = np.array([[0, 0], [src_image.shape[1] - 1, 0], [0, src_image.shape[0] - 1]]).astype(np.float32)
    # so it is know that this matrix is applying a rotation of -50 degrees and 0.6 scale
    dstTri = np.array([[0, src_image.shape[1] * 0.33], [src_image.shape[1] * 0.85, src_image.shape[0] * 0.25],
                       [src_image.shape[1] * 0.15, src_image.shape[0] * 0.7]]).astype(np.float32)

    warp_mat = cv.getAffineTransform(srcTri, dstTri)
    # the matrix is supposed to be:
    # T = S * R where S = [[0.6, 0], [0, 0.6]] and R = [[cos(-50), -sin(-50)],[sin(-50), cos(50)]]

    angle_rad = np.deg2rad(-50)
    cos, sin = np.cos(angle_rad), np.sin(angle_rad)

    S = np.array([[0.6, 0], [0, 0.6]], dtype=np.float32)
    R = np.array([[cos, -sin], [sin, cos]])

    T = S @ R

    warp_dst = cv.warpAffine(src_image, warp_mat, (src_image.shape[1], src_image.shape[0]))

    # Rotating the image after Warp
    center = (warp_dst.shape[1] // 2, warp_dst.shape[0] // 2)
    angle = -50
    scale = 0.6

    rot_mat = cv.getRotationMatrix2D(center, angle, scale)
    warp_rotate_dst = cv.warpAffine(warp_dst, rot_mat, (warp_dst.shape[1], warp_dst.shape[0]))

    cv.imshow('Source image', src_image)
    cv.imshow('Warp', warp_dst)
    cv.imshow('Warp + Rotate', warp_rotate_dst)
    cv.waitKey()



if __name__ == '__main__':
    cat_image_path = os.path.join(os.getcwd(), 'cat_image.jpg')
    main(cat_image_path)
