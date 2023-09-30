import torch
from torch import nn


class pixelBinaryLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, binary_feature_map: torch.Tensor, labels: torch.Tensor):
        assert len(binary_feature_map) == len(labels)
        # expand the labels 
        _, c, w, h = binary_feature_map.shape
        labels = labels.unsqueeze(dim=-1).expand(-1, c * w * h)
        f_bfm, fl= torch.flatten(binary_feature_map), torch.flatten(labels)
        
        assert torch.all((f_bfm >= 0)), "The feature map is supposed to be binary"
        return nn.BCELoss()(f_bfm, fl)
