from torch import nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, 
                kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                eps=1e-05, momentum=0.1, affine=True, norm=True,
                act=nn.ReLU):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels, eps, momentum, affine) if norm else nn.Identity()
        self.act = act() if act else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
