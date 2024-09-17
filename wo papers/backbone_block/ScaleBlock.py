'''
Title
Data-Driven Neuron Allocation for Scale Aggregation Networks

References
http://arxiv.org/abs/1904.09460
https://github.com/Eli-YiLi/ScaleNet/blob/master/pytorch/scalenet.py
'''



import torch
from torch import nn
import torch.nn.functional as F

class ScaleBranch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, up_size=0):
        super(ScaleBranch, self).__init__()
        self.up_size = up_size
        self.max_pool = nn.MaxPool2d(kernel_size)
        # The kernel_size needs to be handled carefully especially when 
        # the spatial size of input features is too small
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if not self.up_size:
            raise ValueError("Please set the upsampling size before forward pass")
        x = self.max_pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = F.interpolate(x, self.up_size, mode="nearest")
        return x


class ScaleBottleneck(nn.Module):
    def __init__(self, in_channels, down_kernel, out_neurons, expand=False):
        super(ScaleBottleneck, self).__init__()

        self.conv_in = nn.Conv2d(in_channels, in_channels, 1 if expand else 3, padding=0 if expand else 1)
        self.bn_in = nn.BatchNorm2d(in_channels)
        self.conv_out = nn.Conv2d(sum(out_neurons), in_channels * 4 if expand else in_channels, 1 if expand else 3, padding=0 if expand else 1)
        self.bn_out = nn.BatchNorm2d(in_channels * 4 if expand else in_channels)

        self.branches = nn.ModuleList(ScaleBranch(in_channels, c_out, k_down) for c_out, k_down in zip(out_neurons, down_kernel))

    def forward(self, x):
        y = self.bn_in(self.conv_in(x)).relu()
        y_ = []
        for branch in self.branches:
            branch.up_size = y.size()[-2:]
            y_.append(branch(y))
        y = torch.cat(y_, dim=1).relu()
        y = self.bn_out(self.conv_out(y))
        return (y + x).relu()

if __name__ == "__main__":
    torch.manual_seed(226)
    t = torch.randn((32, 64, 32, 32))
    scale = ScaleBottleneck(64, [1, 2, 4, 7], [30, 8, 10, 16])
    print(scale.forward(t).size())