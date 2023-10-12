"""
This script contains basic functionalities needed to train Generative Adversarial Networks
"""

from torch import nn

class Discriminator(nn.Module):
    def _disc_block(self, 
                   in_channels, 
                   out_channels, 
                   is_final: bool=False):

        components = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1)]
        # then depending on whether the block is final or not
        if is_final:
            components.append(nn.Tanh())
        else:
            components.extend([nn.BatchNorm2d(), nn.ReLU()])
        
        return nn.Sequential(*components)

    def _build(self, input_channels: int, num_blocks: int):
        
        blocks = []
        c = input_channels
        for _ in range(num_blocks - 1):
            blocks.append(self._disc_block(in_channels=c, out_channels=2 * c, is_final=False))
            c = 2 * c

        # add the final block
        blocks.append(self._disc_block(in_channels=c, out_channels=2*c, is_final=True))

        return nn.Sequential(*blocks)


    def __init__(self, 
                 image_channels: int,
                 num_blocks: int) -> None:
        # build the classifier
        self.discriminator = self._build(input_channels=image_channels, num_blocks=num_blocks, )    
    

class Generator(nn.Module):
    def _gen_block(self, 
                   in_channels: int, 
                   out_channels: int,
                   kernel_size: int = 3,
                   stride: int = 2,
                   is_final: bool = False):
        
        components = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)]
        # then depending on whether the block is final or not
        if is_final:
            components.append(nn.Tanh())
        else:
            components.extend([nn.BatchNorm2d(), nn.ReLU()])
        
        return nn.Sequential(*components)

    def _build(self, input_channels: int, num_blocks: int):
        
        blocks = []
        c = input_channels
        for _ in range(num_blocks - 1):
            blocks.append(self._disc_block(in_channels=c, out_channels=2 * c, is_final=False))
            c = 2 * c

        # add the final block
        blocks.append(self._disc_block(in_channels=c, out_channels=2*c, is_final=True))

        return nn.Sequential(*blocks)


    def __init__(self, 
                 image_channels: int,
                 num_blocks: int) -> None:
        # build the classifier
        self.discriminator = self._build(input_channels=image_channels, num_blocks=num_blocks, )