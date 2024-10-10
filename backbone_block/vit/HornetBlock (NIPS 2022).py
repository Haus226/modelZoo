'''
Title
HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions

References
http://arxiv.org/abs/2207.14284
'''



import torch
from torch import nn
import torch.nn.functional as F
from timm.layers import DropPath

class DWConv(nn.Module):
    def __init__(self, channels, kernel_size, bias):
        super().__init__()
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels, bias=bias)

    def forward(self, x):
        return self.dw_conv(x)

# Proposed in "Global Filter Networks for Image Classification"
# Modified version used
class GlobalLocalFilter(nn.Module):
    def __init__(self, channels, h, w) -> None:
        super().__init__()
        # LayerNorm
        self.pre_norm = nn.GroupNorm(1, channels)
        self.post_norm = nn.GroupNorm(1, channels)
        self.weight = nn.Parameter(torch.rand((channels // 2, h, w, 2)))
        self.dw_conv = DWConv(channels // 2, 3, False)

    def forward(self, x):
        x = self.pre_norm(x)
        x1, x2 = torch.chunk(x, 2, 1)
        B, C, H, W = x2.size()

        x1 = self.dw_conv(x1)
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')
        weight = self.weight
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(weight.permute(3, 0, 1, 2), size=x2.shape[2:4], mode='bilinear', align_corners=True).permute(1, 2, 3, 0)
        weight = torch.view_as_complex(weight.contiguous())
        x2 = x2 * weight
        x2 = torch.fft.irfft2(x2, s=(H, W), dim=(2, 3), norm='ortho')
        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).view(B, 2 * C, H, W)
        x = self.post_norm(x)
        return x

class gnConv(nn.Module):
    def __init__(self, channels, order=5, type="dw", h=14, w=8, alpha=1.0):
        super(gnConv, self).__init__()
        self.order = order
        self.channels = [channels // 2 ** p for p in range(order)]
        self.channels.reverse()
        self.proj_in = nn.Conv2d(channels, 2 * channels, 1)
        self.dw_conv = DWConv(sum(self.channels), 7, True) if type == "dw" else GlobalLocalFilter(sum(self.channels), h, w)
        self.proj_out = nn.Conv2d(channels, channels, 1)

        self.pw_convs = nn.ModuleList(
            [nn.Identity(), *[nn.Conv2d(self.channels[idx], self.channels[idx + 1], 1) for idx in range(order - 1)]]
        )
        self.alpha = alpha

    def forward(self, x):
        x = self.proj_in(x)
        x, x_dw = torch.split(x, (self.channels[0], sum(self.channels)), dim=1)

        x_dw = self.dw_conv(x_dw) / self.alpha
        x_dw_list = torch.split(x_dw, self.channels, dim=1)
        for p in range(self.order):
            x = self.pw_convs[p](x) * x_dw_list[p]
        return self.proj_out(x)
    
class HorNetBlock(nn.Module):
    def __init__(self, channels, p, layer_scale, gn_conv, mlp):
        super(HorNetBlock, self).__init__()
        self.pre_norm = nn.GroupNorm(1, channels)
        self.gn_conv = gn_conv
        self.post_norm = nn.GroupNorm(1, channels)
        self.mlp = mlp

        # Layer scaling proposed in "Going deeper with Image Transformers" (CaiT)
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(channels), 
                                    requires_grad=True) if layer_scale > 0 else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(channels), 
                                    requires_grad=True) if layer_scale > 0 else 1
        self.drop_path = DropPath(p) if p > 0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1.view(x.size(1), 1, 1) * self.gn_conv(self.pre_norm(x)))
        x = x + self.drop_path(self.gamma_2.view(x.size(1), 1, 1) * self.mlp(x))
        return x




if __name__ == "__main__":
    torch.manual_seed(226)
    t = torch.rand((32, 64, 21, 21))
    gn_conv = gnConv(64, 3, "gf", 8, 8, 1.0)
    print(gn_conv(t).size())
    mlp = nn.Sequential(
        nn.Conv2d(64, 256, 1),
        nn.GELU(),
        nn.Conv2d(256, 64, 1)
    )
    hornetblock = HorNetBlock(64, 0.5, 1e-6, gn_conv, nn.Identity())
    print(hornetblock(t).size())
    hornetblock = HorNetBlock(64, 0.5, 1e-6, gn_conv, mlp)
    print(hornetblock(t).size())