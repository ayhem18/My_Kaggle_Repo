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
    

def conv_gw_3_3(x: np.ndarray, w: np.ndarray, upstream_grad: np.ndarray) -> np.ndarray:
    """This function calculates the gradient of the convolution operations with respect to the weight

    Args:
        x (np.ndarray): input: expected to be 
        w (np.ndarray): in

    Returns:
        np.ndarray: The gradient of the convolution operation with respect to the 'weight' parameter
    """

    # check the input really quick
    if x.ndim != 3:
        raise ValueError(f"This function expects 3 dimensional (non batched input). Found: {x.shape}")

    if w.ndim != 3:
        raise ValueError(f"The weight is expected to be 3 dimensional. Found: {w.shape}")

    # make sure the weight shape matches the input
    c1, h, w = x.shape
    c2, k1, k2 = w.shape

    if c1 != c2 or h <= k1 or w <= k2: 
        raise ValueError((f"Please make sure the numbers of channels match, and the width and height of the filter is less or equal to that of the input.\n"
                          f"Found input: {x.shape}, kernel: {w.shape}"))

    output_shape = (h - k1, w - k2)
    # make sure the gradient shape is as expected
    if upstream_grad.shape != output_shape:
        raise ValueError(f"the upstream gradient is expected to be of shape: {output_shape}.\nFound: {upstream_grad.shape}")

    # the function calculates the gradient of the output
    def gradient_matrix(alpha, beta, gamma):
        # alpha represents the parameter's channels,
        # beta:  its height dimension,
        # gamma: its width dimension

        # 'grad' represents the gradient of W_{alpha, beta, gamma} with respect to Z
        grad = np.asarray([[x[alpha, i + beta, j + gamma] for j in range(w - k2 + 1)] for i in range(h - k1 + 1)],
                            dtype=np.float32)

        # make sure 'grad' matches the shape of the output
        assert grad.shape == (h - k1 + 1, w - k2 + 1), "Make sure the grad shape is correct"
        return grad
    
    grads = [gradient_matrix(a, b, g) for a in range(c1) for b in range(k1) for g in range(k2)]

    assert len(grads) == c1 * k1 * k2, "make sure the gradient is taken on all parameters in the weight tensor"

    # the next step is to compute the new array
    grads_array = np.asarray([np.sum(g * upstream_grad) for g in grads])

    # reshape the gradient to account for the weight shape
    return grads_array.reshape(w.shape)


def conv_gw_4_3(x: np.ndarray, w: np.ndarray, upstream_grad: np.ndarray) -> np.ndarray:
    """This function calculates the gradient of the convolution operations with respect to the weight

    Args:
        x (np.ndarray): input: expected to be 
        w (np.ndarray): in

    Returns:
        np.ndarray: The gradient of the convolution operation with respect to the 'weight' parameter
    """

    # check the input really quick
    if x.ndim != 4:
        raise ValueError(f"This function expects 4-dimensional (batched) input. Found: {x.shape}")
    
    if w.ndim != 3:
        raise ValueError(f"The weight is expected to be 3 dimensional. Found: {w.shape}")
    
    # make sure the weight shape matches the input
    batch_size, c1, height, width = x.shape
    c2, k1, k2 = w.shape

    if c1 != c2 or height <= k1 or width <= k2: 
        raise ValueError((f"Please make sure the numbers of channels match, and the width and height of the filter is less or equal to that of the input.\n"
                          f"Found input: {x.shape}, kernel: {w.shape}"))

    output_shape = (batch_size, height - k1 + 1, width - k2 + 1)
    # make sure the gradient shape is as expected
    if upstream_grad.shape != output_shape:
        raise ValueError(f"the upstream gradient is expected to be of shape: {output_shape}.\nFound: {upstream_grad.shape}")


    # the function calculates the gradient of the output
    def gradient_matrix(alpha, beta, gamma):
        # alpha represents the parameter's channels,
        # beta:  its height dimension,
        # gamma: its width dimension

        # 'grad' represents the gradient of W_{alpha, beta, gamma} with respect to Z
        grad = np.asarray([[x[:, alpha, i + beta, j + gamma] for j in range(width - k2 + 1)] for i in range(height - k1 + 1)],
                            dtype=np.float32)
        grad = np.moveaxis(grad, -1, 0)
        # make sure 'grad' matches the shape of the output
        assert grad.shape == (batch_size, height - k1 + 1, width - k2 + 1), "Make sure the grad shape is correct"
        return grad
    
    grads = [gradient_matrix(a, b, g) for a in range(c1) for b in range(k1) for g in range(k2)]

    assert len(grads) == c1 * k1 * k2, "make sure the gradient is taken on all parameters in the weight tensor"

    # the next step is to compute the new array
    grads_array = np.asarray([np.sum(g * upstream_grad) for g in grads])

    # reshape the gradient to account for the weight shape
    return grads_array.reshape(w.shape)
    

def conv_gx_4_3(x: np.ndarray, weight: np.ndarray, upstream_grad: np.ndarray) -> np.ndarray:
    """Compute The gradient of convolutional operation with respect to the input 'x'

    Args:
        x (np.ndarray): input
        w (np.ndarray): weight

    Returns:
        np.ndarray: the gradient of the output with respect to 'x'
    """
    batch_size, _, h, w = x.shape
    c, k1, k2 = weight.shape

    if upstream_grad.shape != (batch_size, h - k1 + 1, w - k2 + 1):
        raise ValueError(f"Expecting the upstream gradient to be of shape: {(batch_size, h - k1 + 1, w - k2 + 1)}")

    def gradient_matrix(k, m, n):
        # k: depth, m: height, n: width
        # the gradient is of the same shape as the output

        grad = np.zeros(shape=(h - k1 + 1, w - k2 + 1), dtype=np.float32)

        # we know that the dz_(i,j) = 0 if m not in [i, i + k1] or n not in[j, j+k2]
        # we can focus on those value for which the gradient is positive
        for i in range(max(0, m - k1 + 1), min(m, h - k1) + 1):
            for j in range(max(0, n - k2 + 1), min(n, w - k2) + 1):
                grad[i, j] = weight[k][m - i][n - j]

        return grad

    grads = [gradient_matrix(a, b, g) for a in range(c) for b in range(h) for g in range(w)]

    grads = np.asarray([np.asarray([np.sum(g * upstream_grad[index_batch, :, :]) for g in grads], dtype=np.float32) for index_batch in range(batch_size)])

    # reshape
    grads = grads.reshape(batch_size, c, h, w)
    
    return grads


def pad(x: np.ndarray, p1: int, p2: int, pad_value: float = 0) -> np.ndarray:
    if x.ndim not in [3, 4]:
        raise ValueError(f"the input is expected to be either 3 or 4 dimensional. Found: {x.shape}")

    if x.ndim == 3:
        new_x = np.pad(x, 
                       [(0, 0), (p1, p1), (p2, p2)], 
                       mode='constant', 
                       constant_values=[(0, 0), (pad_value, pad_value), (pad_value, pad_value)])
        
        # make sure the shape is as expected
        new_shape = (x.shape[0], x.shape[1] + 2 * p1 , x.shape[2] + 2 * p2)

        if new_x.shape != new_shape:
            raise ValueError(f"The result is expected to be shape: {new_shape}.Found: {new_x.shape}")
        
        return new_x

    new_x = np.pad(x, 
                   [(0, 0), (0, 0), (p1, p1), (p2, p2)], 
                   mode='constant', 
                   constant_values=[(0, 0), (0, 0), (pad_value, pad_value), (pad_value, pad_value)])

    new_shape = (x.shape[0], x.shape[1], x.shape[2] + 2 * p1 , x.shape[3] + 2 * p2)

    if new_x.shape != new_shape:
        raise ValueError(f"The result is expected to be shape: {new_shape}.Found: {new_x.shape}")
    
    return new_x


def pad_reverse(x: np.ndarray, p1: int, p2: int) -> np.ndarray:
    """This function given is basically the reverse function of the 'pad' function above. Given y = pad(x, p1, p2, pad_value), 
    this function returns 'x'

    Args:
        x (np.ndarray): The padded
        p1 (int): The number of pads on the height axis
        p2 (int): The number of pads on the width axis

    Returns:
        np.ndarray: the original 'x' before padding
    """
    if x.ndim not in [3, 4]:
        raise ValueError(f"the input is expected to be either 3 or 4 dimensional. Found: {x.shape}")

    if x.ndim == 3:
        org_x = x[:, p1:-p1, p2:-p2]
        org_shape = (x.shape[0], x.shape[1] - 2 * p1 , x.shape[2] - 2 * p2)

        if org_x.shape != org_shape:
            raise ValueError(f"The result is expected to be shape: {org_shape}.Found: {org_x.shape}")
        
        return org_x

    org_x = x[:, :, p1:-p1, p2:-p2]
    org_shape = (x.shape[0], x.shape[1], x.shape[2] - 2 * p1 , x.shape[3] - 2 * p2)

    if org_x.shape != org_shape:
        raise ValueError(f"The result is expected to be shape: {org_shape}.Found: {org_x.shape}")
    
    return org_x


def conv_grad(X: np.ndarray, k: np.ndarray, dL: np.ndarray = None ):
    # Compute backpropagated loss over kernel and input image
    dLdX = np.zeros_like(X, dtype=np.float32)
    dLdK = np.zeros_like(k, dtype=np.float32)

    for i in range(X.shape[0]):
        for j in range(k.shape[0]):
            for h in range(X.shape[2] - k.shape[2] + 1):
                for w in range(X.shape[3] - k.shape[3] + 1):
                    dLdX[i, :, h:h + k.shape[2], w:w + k.shape[3]] += dL[i, j, h, w] * k[j, :, :, :]
                    dLdK[j, :, :, :] += dL[i, j, h, w] * X[i, :, h:h + k.shape[2], w:w + k.shape[3]]
    
    return dLdX, dLdK
