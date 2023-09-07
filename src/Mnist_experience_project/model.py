import torch

from torch import nn
from typing import Union, Tuple, Optional

from src.pytorch_modular.dimensions_analysis.dimension_analyser import DimensionsAnalyser


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

    @classmethod
    def linear_block(cls,
                     input_units,
                     output_units,
                     is_final: bool = False) -> nn.Sequential:
        components = [nn.Linear(in_features=input_units,
                                out_features=output_units)]
        # the rest of the block depends on the is_final variable
        if not is_final:
            components.extend([nn.BatchNorm1d(output_units), nn.LeakyReLU()])

        return nn.Sequential(*components)

    def __init__(self,
                 input_shape: Tuple,
                 num_classes: int,
                 num_conv_blocks: int = 2,
                 min_final_units: int = 16,
                 *args, **kwargs) -> None:
        """

        Args:
            input_shape: the shape of the input
            num_blocks: the number of convolutional blocks in the Model
            num_classes: number of classes: needed for output units
            min_final_units: the dimension of the output of the final convolutional block
        """
        super().__init__(*args, **kwargs)
        self.input_shape = input_shape if len(input_shape) > 2 else input_shape + (1,)
        self.output_units = num_classes if num_classes > 2 else 1
        self.num_conv_blocks = num_conv_blocks
        self.net = self.build_model()

    def build_model(self) -> nn.Sequential:
        # the idea is to have num_blocks convolutional blocks
        # and one linear block
        h, w, c = self.input_shape
        num_channels = 16

        blocks = []
        for i in range(self.num_conv_blocks):
            blocks.append(self.conv_block(input_c=num_channels,
                                          output_c=2 * num_channels,
                                          padding='same', kernel_size=3, stride=1))
            num_channels *= 2

        # append a max pooling layer
        blocks.append(nn.MaxPool2d(kernel_size=2))

        # flatten the output
        blocks.append(nn.Flatten())

        # the next step is to compute the number of units in the output
        temp_net = nn.Sequential(*blocks)
        temp_input_shape = (1,) + self.input_shape
        _, num_units = DimensionsAnalyser().analyse_dimensions(net=temp_net,
                                                               input_shape=temp_input_shape,
                                                               method='static')

        # add the linear block
        blocks.append(self.linear_block(input_units=num_units, output_units=self.output_units, is_final=True))

        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.forward(x)
