'''
Title
ChannelNets: Compact and Efficient Convolutional Neural Networks via Channel-Wise Convolutions

References
https://arxiv.org/abs/1809.01330
'''



import torch
from torch import nn
from utils import ConvBNReLU

class GCWConv(nn.Module):
    def __init__(self, groups, kernel_size, padding):
        super(GCWConv, self).__init__()
        self.conv = nn.Conv3d(1, groups, kernel_size, stride=(groups, 1, 1),
                            padding=padding)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        # The number of output channels will be same as the formula for
        # computing the output spatial size in 2d-Conv, i.e
        # O = (I - K + 2P) / S + 1
        return x.view(x.size(0), -1, x.size(3), x.size(4)).relu()
    
class DWCWConv(nn.Module):
    def __init__(self, channels, kernel_size, padding):
        super(DWCWConv, self).__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels)
        self.cwconv = GCWConv(1, kernel_size, padding)

    def forward(self, x):
        return self.cwconv(self.dwconv(x))

class DWSConv(nn.Module):
    '''
    For implementing ChannelNet
    '''
    def __init__(self, in_channels, out_channels):
        super(DWSConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)

class ChannelBlock(nn.Module):
    '''
    The Group Module(GM) and the Group Channel Wise Module(GCWM) in one class.
    '''
    def __init__(self, in_channels, out_channels, groups, kernels, paddings, cw=True):
        super(ChannelBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, 1, 1, groups=groups, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, 1, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
            GCWConv(groups, kernels, padding=paddings) if cw else nn.Identity()
        )
        self.expand = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()


    def forward(self, x):
        return self.block(x) + self.expand(x)

if __name__ == "__main__":
    torch.manual_seed(226)
    t = torch.rand((32, 64, 21, 21))
    gcw = GCWConv(2, 3, 1)
    print("GCWConv : ", gcw(t).size())
    dwcw = DWCWConv(64, 3, 1)
    print("DWCWConv : ", dwcw(t).size())
    gm = ChannelBlock(64, 512, 4, 3, 1, False)
    print("GM : ", gm(t).size())
    gcwm = ChannelBlock(64, 512, 8, 3, 1, False)
    print("GCWM : ", gcwm(t).size())

