"""
This script is an implementation of the Convolutional Layer
"""
import numpy as np

from typing import Union, Tuple
from .param_layer import ParamLayer
import tinyBackProp.functional.convolution as conv
# from typing import override


class ConvLayer(ParamLayer):
    # the current version of the layer only works with 1 output channel
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[Tuple[int, int], int],
                 padding: Union[Tuple[int, int], int] = None,
                 weight_matrix: np.ndarray = None):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel_size if isinstance(kernel_size, Tuple) else (kernel_size, kernel_size)
        self.padding = padding if (isinstance(padding, (tuple)) or padding is None) else (padding, padding)

        if weight_matrix is not None:
            # expand the weight matrix is needed
            weight_matrix = np.expand_dims(weight_matrix, axis=0) if weight_matrix.ndim == 3 else weight_matrix
            # make sure the input is of the correct shape
            exp_shape = (self.out_channels, self.in_channels, self.kernel[0], self.kernel[1])
            if weight_matrix.shape != exp_shape:
                raise ValueError(f"Please make sure the weight matrix is of the expected shape\n"
                                 f"Expected: {exp_shape}. Found: {weight_matrix.shape}")

        # set the field
        self.weight = np.random.rand(self.out_channels, self.in_channels, self.kernel[0],
                                     self.kernel[1]) if weight_matrix is None else weight_matrix

    # @override
    def _verify_input(self, x: np.ndarray) -> np.ndarray:
        # make sure the input is either 2 or 3-dimensional
        if x.ndim not in [3, 4]:
            raise ValueError(f"The input is expected to be either 3 or 4 dimensional. Found: {x.ndim} dimensions")

        x = np.expand_dims(x, axis=0) if x.ndim == 3 else x.copy()
        # extract the shape
        _, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"The input is expected to have {self.in_channels} channels")

        _, c, k1, k2 = self.weight.shape

        if h <= k1 or w <= k2:
            raise ValueError(f"The dimensions of the input are expected to be larger than those of the kernel\n"
                             f"input dimensions:{(h, w)}, kernel dimensions: {k1, k2}")

        return x

    def forward(self, x: np.ndarray = None):
        
        
        # apply padding if needed
        if self.padding is not None:
            x = conv.pad(x, p1=self.padding[0], p2=self.padding[1], pad_value=0)

        x = self._verify_input(x)
        # extract the dimensions of the image
        batch_size, _, h, w = x.shape
        # extract the dimensions of the weight
        _, _, k1, k2 = self.weight.shape

        # create the feature map to save the result
        feature_map = np.zeros(shape=(batch_size, self.out_channels, h - k1 + 1, w - k2 + 1))

        for index_channel in range(self.out_channels):
            for y_corner in range(h - k1 + 1):
                for x_corner in range(w - k2 + 1):
                    product_matrix = x[:, :, y_corner: y_corner + k1, x_corner: x_corner + k2] * self.weight[index_channel]
                    feature_map[list(range(batch_size)), index_channel, y_corner, x_corner] = np.sum(np.reshape(product_matrix, (batch_size, -1)), axis=1)
 

        expected_shape = (batch_size, self.out_channels, h - k1 + 1, w - k2 + 1)
        if feature_map.shape != expected_shape:
            raise ValueError((f"Expected output's shape: {expected_shape}\n"
                              f"Found: {feature_map.shape}"))

        return feature_map


    def param_grad(self, x: np.ndarray = None, upstream_grad: np.ndarray = None) -> np.ndarray: 
        # make sure to pad first if needed
        x_pad = conv.pad(x, p1=self.padding[0], p2=self.padding[1], pad_value=0) if self.padding is not None else x

        gradient = np.asarray([conv.conv_gw_4_3(x=x_pad, 
                                                w=(self.weight[i, :, :, :]), 
                                                upstream_grad=(upstream_grad[:, i, : , :])) 
                            
                            for i in range(self.out_channels)])

        if gradient.shape != (self.out_channels, self.in_channels, self.kernel[0], self.kernel[1]):
            raise ValueError(f"The gradient shape does not match the weight")
        
        return gradient


    def grad(self, x:np.ndarray = None, upstream_grad: np.ndarray = None) -> np.ndarray:
        # pad if needed
        x_pad = conv.pad(x, p1=self.padding[0], p2=self.padding[1], pad_value=0) if self.padding is not None else x

        gradients = np.zeros(x_pad.shape)
        for i in range(self.out_channels):
            gradients += conv.conv_gx_4_3(x_pad, self.weight[i], upstream_grad=upstream_grad[:, i, : , :])
        
        if self.padding is not None:
            gradients = conv.pad_reverse(gradients, self.padding[0], self.padding[1])
            if gradients.shape != x.shape:
                raise ValueError((f"The shape of the final gradient is different from that of the input\n"
                                  f"Gradient: {gradients.shape}. x: {x.shape}"))

        return gradients



    # @override
    def local_param_grads(self, x: np.ndarray) -> list[np.ndarray]:
        """
        This function calculates the derivative of the output with respect to every scalar in the weight tensor.
        Args:
            x: The input

        Returns: A list of numpy arrays of the same shape as the output, where each element in the list
        represents the derivative of the output with respect to the
        """

        # extract the dimensions of the input
        _, h, w = x.shape
        # extract the dimensions of the weights
        _, c, k1, k2 = self.weight.shape

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

        # calculate the gradient for each parameter in the weights tensor
        grads = [gradient_matrix(a, b, g) for a in range(c) for b in range(k1) for g in range(k2)] * self.out_channels

        assert len(grads) == self.out_channels * c * k1 * k2, "make sure the gradient is taken on all parameters in the weight tensor"
        return grads


    def local_x_grad(self, x: np.ndarray):
        """
        This function generates the gradient of the output with respect to the input.
        Args:
            x: the input
        Returns: a List of numpy arrays. Each with the same shape as the input representing the gradient
        of the output with respect to an entry in 'x'
        """

        _, h, w = x.shape
        c, k1, k2 = self.weight.shape

        def gradient_matrix(k, m, n):
            # k: depth, m: height, n: width
            
            # the gradient is of the same shape as the output

            grad = np.zeros(shape=(h - k1 + 1, w - k2 + 1), dtype=np.float32)

            # we know that the dz_(i,j) = 0 if m not in [i, i + k1] or n not in[j, j+k2]
            # we can focus on those value for which the gradient is positive
            for i in range(max(0, m - k1 + 1), min(m, h - k1) + 1):
                for j in range(max(0, n - k2 + 1), min(n, w - k2) + 1):
                    grad[i][j] = self.weight[k][m - i][n - j]

            return grad

        grads = [gradient_matrix(a, b, g) for a in range(c) for b in range(h) for g in range(w)]

        assert len(grads) == c * h * w, "make sure the gradient is taken on all parameters in the weight tensor"
        return grads
