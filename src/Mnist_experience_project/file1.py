"""
This script is my main attempt to reach 100% accuracy on the Kaggle Mnist Dataset (test split of course xD)
"""

from torch import nn
from typing import Union, Tuple


class BaselineModel(nn.Module):
    @classmethod
    def conv_block(cls,
                   input_c: int,
                   output_c: int,
                   kernel_size: Union[int, Tuple[int, int]] = 3,
                   stride: Union[int, Tuple[int, int]] = 1,
                   padding: Union[str, int] = 'same') -> nn.Sequential:
        # let's start with something interesting here
        return nn.Sequential(nn.Conv2d(input_c,
                                       output_c,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding
                                       ),
                             nn.BatchNorm2d(output_c),
                             # let's keep it simple use the default scope
                             nn.LeakyReLU())

    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 num_classes: int,
                 num_blocks: int = 2,
                 min_final_units: int = 16,
                 *args, **kwargs) -> None:
        """

        Args:
            input_shape: the shape of the input
            num_blocks: the number of convolutional blocks in the Model
            num_classes: number of classes: needed for output units
            min_final_units: the dimension of the output of the final convolutional block
            *args:
            **kwargs:
        """
        super().__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.output_units = num_classes if num_blocks > 2 else 1
        self.num_blocks = num_blocks

