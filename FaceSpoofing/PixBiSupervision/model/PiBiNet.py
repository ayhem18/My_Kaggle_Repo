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
                 feature_map_channels: int = 5,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        # first initialize the backbone
        self.backbone = DenseNetFeatureExtractor(2, minimal=True)
        # add the 1 * 1 convolutional network
        self.conv = nn.Conv2d(in_channels=384,
                              out_channels=feature_map_channels,
                              kernel_size=(1, 1), )
        self.sigmoid_layer = nn.Sigmoid()
        self.linear = nn.Linear(in_features=14 * 14 * feature_map_channels, out_features=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # first make sure the input is of the expected dimensions
        assert len(x.shape()) == 4, "The input is expected to be 4 dimensional"
        # make sure the height and width are set to the expected dimensions: (244, 244)
        assert x.shape[1:] == (3, 224, 224), (f"The picture are expected to be of dimensions: {3, 244, 244}"
                                              f"\n Found: {x.shape[1:]}")

        # pass the input through the backbone, convolution and sigmoid layer
        feature_map = self.sigmoid_layer(self.conv(self.backbone(x)))

        # flatten, then pass to the linear and sigmoid layer again
        # label = self.sigmoid_layer(self.linear(nn.Flatten(feature_map)))
        label = self.linear(nn.Flatten(feature_map))

        return feature_map, label

