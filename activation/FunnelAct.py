'''
Title
Funnel Activation for Visual Recognition

References
http://arxiv.org/abs/2007.11824
'''



import torch
from torch import nn

class FunnelReLU(nn.Module):
    def __init__(self, channels, kernel_size):
        super(FunnelReLU, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2, groups=channels)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return torch.max(x, self.bn(self.conv(x)))