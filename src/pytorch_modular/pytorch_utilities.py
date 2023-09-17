"""
Unlike helper_functionalities.py, this script contains, Pytorch code that is generally used across different
scripts and Deep Learning functionalities
"""

import torch
from torch import nn
from typing import Union
from torch.utils.data import DataLoader
from pathlib import Path
import os
from datetime import datetime as d
from src.pytorch_modular.directories_and_files import process_save_path

HOME = os.getcwd()


# set the default device
def get_default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_module_device(module: nn.Module) -> str:
    # this function is mainly inspired by this overflow post:
    # https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently
    if hasattr(module, 'device'):
        return module.device
    return next(module.parameters()).device



# def __dimensions_after_conv(h: int, w: int, conv: nn.Conv2d) -> tuple[int, int]:
#     # this code is based on the documentation of conv2D module pytorch:
#     # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

#     # extract the numerical features first
#     s1, s2 = conv.stride
#     k1, k2 = conv.kernel_size
#     d1, d2 = conv.dilation

#     # the padding is tricky
#     if conv.padding == 'same':
#         return h, w

#     if conv.padding == 'valid':
#         p1, p2 = 0, 0

#     else:
#         p1, p2 = conv.padding

#     new_h = int((h + 2 * p1 - d1 * (k1 - 1) - 1) / s1) + 1
#     new_w = int((w + 2 * p2 - d2 * (k2 - 1) - 1) / s2) + 1

#     return new_h, new_w


# def __dimensions_after_pool(h: int, w: int, pool: Union[nn.MaxPool2d, nn.AvgPool2d]) -> tuple[int, int]:
#     # extract the kernel_size values first
#     kernel = pool.kernel_size if isinstance(pool.kernel_size, tuple) else (pool.kernel_size, pool.kernel_size)
#     k1, k2 = kernel
#     # extract the stride values second
#     stride = pool.stride if isinstance(pool.stride, tuple) else (pool.stride, pool.stride)
#     s1, s2 = stride
#     return int((h - k1) / s1) + 1, int((w - k2) / s2) + 1


# def dimensions_block(input_shape: Union[tuple[int, int], int], block: nn.Sequential) -> tuple[int, int]:
#     # first extract the initial input shape
#     input_h, input_w = input_shape if isinstance(input_shape, tuple) else (input_shape, input_shape)
#     # iterate through the layers of the block and modify the shape accordingly depending on the layer
#     for layer in block:
#         if isinstance(layer, nn.Conv2d):
#             input_h, input_w = __dimensions_after_conv(input_h, input_w, layer)
#         elif isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d)):
#             input_h, input_w = __dimensions_after_pool(input_h, input_w, layer)

#     return input_h, input_w


# def input_shape_from_dataloader(data_loader: torch.utils.data.DataLoader):
#     # first convert to an iterator
#     batch = next(iter(data_loader))

#     # if the data loader returns a tuple, then it is usually batch of image and a batch of labels
#     if isinstance(batch, tuple):
#         # separate images and labels
#         batch_images, _ = batch
#         if isinstance(batch_images[0], torch.Tensor):
#             return tuple(batch_images[0].shape)
#         else:
#             return batch_images[0].shape

#     # this generally considers the case of data loader for test data
#     else:
#         if isinstance(batch[0], torch.Tensor):
#             return tuple(batch[0].shape)
#         else:
#             return batch[0].shape


def __verify_extension(p):
    return os.path.basename(p).endswith('.pt') or os.path.basename(p).endswith('.pth')


def save_model(model: nn.Module, path: Union[str, Path] = None) -> None:
    # the time of saving the model
    now = d.now()
    file_name = "-".join([str(now.month), str(now.day), str(now.hour), str(now.minute)])
    # add the extension
    file_name += '.pt'

    # first check if the path variable is None:
    path = path if path is not None else os.path.join(HOME, file_name)

    # process the path
    path = process_save_path(path,
                             dir_ok=True,
                             file_ok=True,
                             condition=lambda p: not os.path.isfile(p) or __verify_extension(p),
                             error_message='MAKE SURE THE FILE PASSED IS OF THE CORRECT EXTENSION')

    if os.path.isdir(path):
        path = os.path.join(path, file_name)

    # finally save the model.
    torch.save(model.state_dict(), path)


def load_model(base_model: nn.Module,
               path: Union[str, Path]) -> nn.Module:
    # first process the path
    path = process_save_path(path,
                             dir_ok=False,
                             file_ok=True,
                             condition=lambda p: not os.path.isfile(p) or __verify_extension(p),
                             error_message='MAKE SURE THE FILE PASSED IS OF THE CORRECT EXTENSION')

    base_model.load_state_dict(torch.load(path))

    return base_model
