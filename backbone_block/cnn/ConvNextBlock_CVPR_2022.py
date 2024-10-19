'''
Title
A ConvNet for the 2020s

References
http://arxiv.org/abs/2201.03545
https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
'''



import torch
from torch import nn
from timm.layers import DropPath

class ConvNextBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, layer_scale=1e-6, p=0):
        super(ConvNextBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 7, padding=3, groups=in_channels),
            nn.GroupNorm(1, in_channels),
            # If LayerNoem is used, the conv layer can be replaced by linear layer
            # But remember to permute the input into BHWC
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.GELU(),
            nn.Conv2d(mid_channels, in_channels, 1)
        )
        self.layer_scale = nn.Parameter(layer_scale * torch.ones((in_channels, 1, 1)))
        # DropPath used in first reference code (official) while StochasticDepth used in second
        # self.drop = StochasticDepth(p)
        self.drop = DropPath(p)

    def forward(self, x):
        return self.drop(self.layer_scale * self.block(x)) + x
    
if __name__ == "__main__":
    torch.manual_seed(226)
    t = torch.rand(32, 64, 21, 21)
    convnext = ConvNextBlock(64, 256, 128)
    print(convnext(t).size())