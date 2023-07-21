"""
This script contains functionalities to build classifier on top of the pretrained model
'RESNET 50' provided by pytorch module

This script is mainly inspired by this paper: 'https://arxiv.org/abs/1411.1792'
I use the main framework suggested by the paper the portion of layers with most transferable 'general' layers
"""
import os
import sys

HOME = os.getcwd()
sys.path.append(HOME)
sys.path.append(os.path.join(HOME, 'src'))

from torchvision.models import resnet50, ResNet50_Weights
from torch import nn

LAYER_BLOCK = 'channel'
RESIDUAL_BLOCK = 'residual'


# noinspection PyUnresolvedReferences
class feature_extractor(nn.Module):
    # the model's architecture refers to blocks with the same number of channels as 'layers'

    def __max_net_layers(self):
        return sum([isinstance(m, nn.Sequential) for m in self.net.modules()])

    def __feature_extractor_layers(self, number_of_layers, freeze: bool):
        modules_generator = self.net.modules()
        # first let's remove the first element
        next(modules_generator)

        def next_module():
            counter = 0
            for module in modules_generator:
                if not isinstance(module, nn.Sequential) or counter < number_of_layers:
                    yield module
                    counter += 1

        modules_to_keep = [m for m in next_module()]
        fe = nn.Sequential(modules_to_keep)
        if freeze:
            for p in fe.parameters():
                p.requires_grad = False

        return fe

    def __init__(self, blocks_to_keep: int,  # the number of blocks to keep
                 blocks_type: str = LAYER_BLOCK,  # the type of blocks to consider (layers or residual blocks)
                 freeze: bool = True,  # whether to freeze the chosen layers or not
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        # first make sure the blocks_type argument is correct
        assert blocks_type in [LAYER_BLOCK, RESIDUAL_BLOCK]

        self.net = resnet50
        self.feature_extractor = None

        if blocks_type == LAYER_BLOCK:
            self.feature_extractor = self.__feature_extractor_layers(blocks_to_keep, freeze=freeze)
        else:
            pass


if __name__ == '__main__':
    default_weights = ResNet50_Weights.DEFAULT
    net = resnet50(default_weights)
    for c in net.modules():
        # print(n)
        print("#" * 100)
