"""
This script offers a number of functionalities to optimize the convolutional operation depending on different scenarios
"""
import numpy as np


def conv_3_3(x: np.ndarray, weight: np.ndarray):
    # this function performs the convolution operation at its most basic form
    # with input 'x' and 'w' as 3 dimensional entities

    if x.ndim not in [2, 3]:
        raise ValueError(f"the input is expected to be either 2 or 3 dimensional\n Found: {x.shape}")

    if weight.ndim not in [2, 3]:
        raise ValueError(f"the input is expected to be either 2 or 3 dimensional\n Found: {weight.shape}")

    # unsqueeze if needed
    x = np.expand_dims(x, axis=0) if x.ndim == 2 else x
    weight = np.expand_dims(weight, axis=0) if weight.ndim == 2 else weight

    c1, h, w = x.shape    
    # extract the dimensions of the weight
    c2, k1, k2 = weight.shape

    if c1 != c2: 
        raise ValueError(f"expected both input and weight to have the same outer dimension\nFound, x: {x.shape} and w: {weight.shape}")



    # create the feature map to save the result
    feature_map = np.zeros(shape=(h - k1 + 1, w - k2 + 1))

    for y_corner in range(h - k1 + 1):
        for x_corner in range(w - k2 + 1):
            feature_map[y_corner, x_corner] = np.sum(
                x[:, y_corner: y_corner + k1, x_corner: x_corner + k2] * weight)

    return feature_map


def conv_4_3(x: np.ndarray, weight: np.ndarray):
    # this function performs the convolution operation with batched input
    # with input 'x' and 'w' as 3 dimensional entities

    if x.ndim not in [3, 4]:
        raise ValueError(f"the input is expected to be either 3 or 4 dimensional\n Found: {x.shape}")

    if weight.ndim not in [2, 3]:
        raise ValueError(f"the input is expected to be either 2 or 3 dimensional\n Found: {weight.shape}")

    # unsqueeze if needed
    x = np.expand_dims(x, axis=0) if x.ndim == 2 else x
    weight = np.expand_dims(weight, axis=0) if weight.ndim == 2 else weight

    batch_size, c1, h, w = x.shape    
    # extract the dimensions of the weight
    c2, k1, k2 = weight.shape

    if c1 != c2: 
        raise ValueError(f"expected both input and weight to have the same outer dimension\nFound, x: {x.shape} and w: {weight.shape}")

    # create the feature map to save the result
    feature_map = np.zeros(shape=(batch_size, h - k1 + 1, w - k2 + 1))

    for y_corner in range(h - k1 + 1):
        for x_corner in range(w - k2 + 1):
            # since weight is 3 dimensional, it will be broadcasted
            # and each element in the batched 'x' will be multiplicated by the same weight
            product_matrix = x[:, :, y_corner: y_corner + k1, x_corner: x_corner + k2] * weight
            
            # The idea here is that we are finding the value at the index(x_corner, y_corner) 
            # product_matrix is 4 dimensional, we will need each outer dimension to be associated with the sum of the other dimensions
            # (the result of the convolution operation on (3, 3) operands)
            # reshape and sum over the first dimension. 
            feature_map[list(range(batch_size)), y_corner, x_corner] = np.sum(np.reshape(product_matrix, (batch_size, -1)), axis=1)

    return feature_map
    
    

