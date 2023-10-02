"""
This is an implementation of the Face Spoofing network suggested in the paper
'Deep Pixel-wise Binary Supervision for Face Presentation Attack Detection'
"""
import torch
from torch import nn

from FaceSpoofing.PixBiSupervision.model.DenseNetFE import DenseNetFeatureExtractor  

class PiBiNet(nn.Module):
    default_transformation = DenseNetFeatureExtractor.default_transform

    def __init__(self,
                 num_blocks: int = 4, 
                 feature_map_channels: int = 5,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        # first initialize the backbone
        self.backbone = DenseNetFeatureExtractor(num_blocks=num_blocks, minimal=True, freeze=True)
        # add the 1 * 1 convolutional network
        self.conv = nn.Conv2d(in_channels=512,
                              out_channels=feature_map_channels,
                              kernel_size=(1, 1), )
        
        self.sigmoid_layer = nn.Sigmoid()

        self.linear = nn.Linear(in_features=125, out_features=1)
        self.model = nn.Sequential(self.backbone, 
                                   self.conv, 
                                   self.sigmoid_layer, 
                                   nn.Flatten(), 
                                   self.linear)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # first make sure the input is of the expected dimensions
        assert len(x.shape) == 4, "The input is expected to be 4 dimensional"
        # make sure the height and width are set to the expected dimensions: (244, 244)
        assert x.shape[1:] == (3, 160, 160), (f"The picture are expected to be of dimensions: {(3, 160, 160)}"
                                              f"\n Found: {x.shape[1:]}")

        # return self.model.forward(x)

        # pass the input through the backbone, convolution and sigmoid layer
        feature_map = self.sigmoid_layer(self.conv(self.backbone(x)))

        # flatten, then pass to the linear and sigmoid layer again
        # label = self.sigmoid_layer(self.linear(nn.Flatten(feature_map)))
        logits = self.linear(nn.Flatten()(feature_map))

        return feature_map, logits
