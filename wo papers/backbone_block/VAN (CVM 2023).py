'''
Title
Visual Attention Network

References
http://arxiv.org/abs/2202.09741
'''



import torch
from torch import nn
from timm.layers import DropPath

class LKABlock(nn.Module):
    def __init__(self, channels):
        super(LKABlock, self).__init__()
        self.dw_conv = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.ddw_conv = nn.Conv2d(channels, channels, 7, padding=9, dilation=3, groups=channels)
        self.pw_conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        return x * self.pw_conv(self.ddw_conv(self.dw_conv(x)))
    
class Attention(nn.Module):
    def __init__(self, channels):
        super(Attention, self).__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            LKABlock(channels),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, x):
        return self.attn(x) + x
    
class VANBlock(nn.Module):
    def __init__(self, channels, drop_path,
                mlp=nn.Identity, 
                layer_scale=1e-2,
                act_layer=nn.GELU):
        super(VANBlock, self).__init__()
        self.bn_1 = nn.BatchNorm2d(channels)
        self.attn = Attention(channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0  else nn.Identity()

        self.bn_2 = nn.BatchNorm2d(channels)
        self.mlp = mlp()
        

        self.layer_scale_1 = nn.Parameter(layer_scale * torch.ones(channels)) if layer_scale > 0 else 1.0
        self.layer_scale_2 = nn.Parameter(layer_scale * torch.ones(channels)) if layer_scale > 0 else 1.0

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.view(x.size(1), 1, 1) * self.attn(self.bn_1(x)))
        x = x + self.drop_path(self.layer_scale_1.view(x.size(1), 1, 1) * self.mlp(self.bn_2(x)))
        return x
    
if __name__ == "__main__":
    torch.manual_seed(226)
    t = torch.rand((32, 64, 21, 21))
    van = VANBlock(64, 0.5)
    print(van(t).size())


        
        
    