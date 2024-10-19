'''
Title
Deformable Convolutional Networks
Deformable ConvNets v2: More Deformable, Better Results

References
http://arxiv.org/abs/1703.06211
https://arxiv.org/abs/1811.11168
https://github.com/4uiiurz1/pytorch-deform-conv-v2/blob/master/deform_conv_v2.py
'''



import torch
from torch import nn
from einops import rearrange

class DeformConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=None, modulation=True):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable Convself.Nets v2).
        """
        super(DeformConv, self).__init__()
        self.kernel_size = kernel_size
        self.N = kernel_size * kernel_size
        self.padding = padding
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.offset_conv = nn.Conv2d(in_channels, 2 * self.N, kernel_size=kernel_size, padding=padding, stride=stride)

        self.modulation = modulation
        if modulation:
            self.modulation_conv = nn.Conv2d(in_channels, self.N, kernel_size=kernel_size, padding=padding, stride=stride)


    def forward(self, x):
        b, c, h, w = x.size()
        offset = self.offset_conv(x)

        # (B, 2N, H', W')
        p = self._get_p(offset)
        h_, w_ = p.shape[2:]


        p = p.permute(0, 2, 3, 1)
        p = torch.cat([torch.clamp(p[..., :self.N], 0, h - 1), torch.clamp(p[..., self.N:], 0, w - 1)], dim=-1)

        p[..., :self.N] = 2.0 * p[..., :self.N] / max(h - 1, 1) - 1.0  
        p[..., self.N:] = 2.0 * p[..., self.N:] / max(w - 1, 1) - 1.0  

        p = p.view(b, h_, w_, 2, self.kernel_size * self.kernel_size)
        p = p.permute(0, 1, 2, 4, 3).reshape(b, h_ * self.kernel_size, w_ * self.kernel_size, 2)
        x_offset = torch.nn.functional.grid_sample(x, p, align_corners=True).transpose(-2, -1)

        # Deformable Convolution V2
        if self.modulation:
            m = torch.sigmoid(self.modulation_conv(x))
            m = m.permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset = rearrange(x_offset, "b c (h k1) (w k2) -> b c h w (k1 k2)", 
                                k1=self.kernel_size, k2=self.kernel_size) * m
            x_offset = rearrange(x_offset, "b c h w (k1 k2) -> b c (h k1) (w k2)", k1=self.kernel_size, k2=self.kernel_size)

        y = self.conv(x_offset)
        return y

    def _get_p(self, offset):
        h, w = offset.size(2), offset.size(3)
        # Get the relative coordinates respected to the center pixel 
        p_n_y, p_n_x = torch.meshgrid(
            torch.arange(-(self.kernel_size // 2), self.kernel_size // 2 + 1),
            torch.arange(-(self.kernel_size // 2), self.kernel_size // 2 + 1),
            indexing="ij"
        )
        p_n = torch.cat([torch.flatten(p_n_y), torch.flatten(p_n_x)])
        p_n = p_n.view(1, 2 * self.N, 1, 1)

        # Get the indices of the center pixels
        p_0_h, p_0_w = torch.meshgrid(
            torch.arange(self.kernel_size // 2,  h * self.stride + 1, self.stride),
            torch.arange(self.kernel_size // 2,  w * self.stride + 1, self.stride),
            indexing="ij"
        )
        p_0_h = p_0_h.view(1, 1, h, w).repeat(1, self.N, 1, 1)
        p_0_w = p_0_w.view(1, 1, h, w).repeat(1, self.N, 1, 1)
        p_0 =  torch.cat([p_0_h, p_0_w], 1)

        p = p_0.to(offset.device) + p_n.to(offset.device) + offset
        return p