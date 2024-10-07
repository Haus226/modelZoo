'''
Title
InceptionNeXt: When Inception Meets ConvNeXt

References
http://arxiv.org/abs/2303.16900
'''



import torch
from torch import nn
from utils import ConvBNReLU
from timm.layers import DropPath

class InceptionNextBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=5, band_size=11, channels_ratio=0.125):
        super(InceptionNextBlock, self).__init__()
        ratio = int(in_channels * channels_ratio)
        self.conv_hw = ConvBNReLU(ratio, ratio, kernel_size, padding=kernel_size // 2, groups=ratio, norm=False, relu=False)
        self.conv_w = ConvBNReLU(ratio, ratio, (1, band_size), padding=(0, band_size // 2), groups=ratio, norm=False, relu=False) 
        self.conv_h = ConvBNReLU(ratio, ratio, (band_size, 1), padding=(band_size // 2, 0), groups=ratio, norm=False, relu=False) 
        self.splits = [ratio, ratio, ratio, in_channels - 3 * ratio]

    def forward(self, x):
        x_hw, x_w, x_h, x_id = torch.split(x, self.splits, dim=1)
        return torch.cat([self.conv_hw(x_hw), self.conv_w(x_w), self.conv_h(x_h), x_id], dim=1)
    
# A general structure abstracted from ConvNext just like MetaFormer
class MetaNextBlock(nn.Module):
    def __init__(self, channels, token_mixer, norm_layer, mlp_layer, act_layer, layer_scale=1e-6, p=0):
        super(MetaNextBlock, self).__init__()
        self.token_mixer = token_mixer(channels)
        self.norm = norm_layer(channels)
        # Mlp should equipped with activation layer
        self.mlp = mlp_layer(channels, act_layer)
        self.layer_scale = nn.Parameter(layer_scale * torch.ones(channels, 1, 1)) if layer_scale else 1
        self.drop = DropPath(p) if p else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.mlp(self.norm(self.token_mixer(x)))
        return self.drop(self.layer_scale * x) + shortcut

if __name__ == "__main__":
    torch.manual_seed(226)
    t = torch.rand((32, 64, 21, 21))
    inceptionnext = InceptionNextBlock(64, 5, 11, 0.125)
    print(inceptionnext(t).size())
    metanext = MetaNextBlock(64, InceptionNextBlock, nn.BatchNorm2d, nn.Identity, nn.ReLU)
    print(metanext(t).size())