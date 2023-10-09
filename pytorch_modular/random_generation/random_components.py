"""
This script contains functionalities to randomly generates different frequently used components
such as convolutional, linear and pooling layers.

THIS SCRIPT WILL COME QUITE HANDY FOR TESTING PURPOSES
"""
import random
from torch import nn
from random import randint as ri
from typing import Union, Tuple

random.seed(69)


def random_conv2D(in_channels: int = None,
                  out_channels: int = None) -> nn.Conv2d:
    if in_channels is None:
        in_channels = ri(3, 100)

    if out_channels is None:
        out_channels = ri(5, 100)

    # first let's randomly decide the padding
    p = random.random()
    if p < 0.1:
        padding = 'valid'
    elif p < 0.6:
        padding = 'same'
    elif p < 0.8:
        padding = ri(1, 5)
    else:
        padding = (ri(1, 5), ri(1, 5))

    # set the kernel size
    # even kernel raises warning: let's stick to odd ones.
    p = random.random()
    if p < 0.5:
        n = ri(1, 5)
        kernel = n + int(n % 2 == 0)
    else:
        n1, n2 = ri(1, 5), ri(1, 5)
        kernel = (n1 + int(n1 % 2 == 0), n2 + int(n2 % 2 == 0))

    # strides larger than 1 are not supported with 'same' padding
    if padding == 'same':
        return nn.Conv2d(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel,
                         padding=padding)

    # STRIDES MUST BE STRICTLY LARGER THAN 0
    # set the stride
    p = random.random()
    if p < 0.5:
        stride = ri(1, 3)
    else:
        stride = (ri(1, 3), ri(1, 3))

    # set the dilation
    p = random.random()
    if p < 0.5:
        dilation = ri(1, 3)
    else:
        dilation = (ri(1, 3), ri(1, 3))

    return nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel,
                     stride=stride,
                     dilation=dilation,
                     padding=padding)


def random_pool_layer(pool_type: str = None) -> Union[nn.AvgPool2d, nn.MaxPool2d]:
    if pool_type is None:
        p = random.random()
        pool_type = 'avg' if p < 0.5 else 'max'

    if pool_type not in ['avg', 'max']:
        raise ValueError("the 'pool_type' argument is expected to be either 'avg' or 'max'\n"
                         f"Found: {pool_type}")

    # set the kernel size
    p = random.random()
    if p < 0.5:
        kernel = ri(2, 4)
    else:
        kernel = (ri(2, 4), ri(2, 4))

    p = random.random()
    if p < 0.5:
        padding = ri(0, 1)
    else:
        padding = (ri(0, 1), ri(0, 1))

    # set the stride
    p = random.random()
    if p < 0.5:
        stride = ri(1, 3)
    else:
        stride = (ri(1, 3), ri(1, 3))

    # set the dilation
    p = random.random()
    if p < 0.5:
        dilation = ri(1, 3)
    else:
        dilation = (ri(1, 3), ri(1, 3))

    if pool_type == 'avg':
        layer = nn.AvgPool2d(kernel_size=kernel)
    else:
        layer = nn.MaxPool2d(kernel_size=kernel)

    layer.stride = stride
    layer.padding = padding

    # NOTE: only maxPool2d supports dilation
    if pool_type == 'max':
        layer.dilation = dilation

    return layer


def random_linear_layer() -> nn.Linear:
    return nn.Linear(in_features=ri(10, 1000), out_features=ri(10, 1000))


def random_conv_block(in_channels: int = None,
                      out_channels: int = None,
                      return_in_c: bool = True) -> Union[Tuple[int, nn.Sequential], nn.Sequential]:
    # the convolutional block will start with random convolutional layer
    conv1 = random_conv2D(in_channels, out_channels)
    in_channels = conv1.in_channels
    out_c = conv1.out_channels

    num_layers = ri(3, 12)
    module = [conv1]
    for _ in range(num_layers):
        p = random.random()
        # pooling layers decreases the spatial dimensions significantly: Thus, they should be less frequent
        # in the architecture of the convolutional block
        if p < 0.1:
            layer = random_pool_layer()
        else:
            layer = random_conv2D(in_channels=out_c)
            out_c = layer.out_channels
        module.append(layer)

    # convert the list to a nn.Sequential module: Don't forget to unzip the list
    module = nn.Sequential(*module)
    if return_in_c:
        return in_channels, module

    return module
