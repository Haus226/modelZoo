'''
Title
Involution: Inverting the Inherence of Convolution for Visual Recognition

References
http://arxiv.org/abs/2103.06255
'''



import torch
from torch import nn

class Involution(nn.Module):
    def __init__(self, input_channels, kernel_size, r, stride, groups):
        super(Involution, self).__init__()
        self.gap = None
        if stride > 1:
            self.gap = nn.AvgPool2d((stride, stride))
        self.conv1 = nn.Conv2d(input_channels, input_channels // r, 1)
        self.bn = nn.BatchNorm2d(input_channels // r)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(input_channels // r, kernel_size * kernel_size * groups, 1)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size - 1) // 2, stride)
        self.G = groups
        self.input_channels = input_channels
        self.group_channels = input_channels // self.G
        self.K = kernel_size

    def forward(self, x):
        # Try to draw an example to better understand the concepts of
        # Channel-agnostic and Spatial Specific
        weight = self.conv2(self.relu(self.bn(self.conv1(self.gap(x) if self.gap else x))))
        B, C, H, W = weight.size()
        # The 3-th axis is the flattened K * K neighbor centered at (i, j) pixels where
        # 0 <= i <= H - 1 and 0 <= W <= W - 1 (Need to pad the spatial features)
        x_unfold = self.unfold(x).view(B, self.G, self.group_channels, self.K * self.K, H, W)
        weight = weight.view(B, self.G, 1, self.K * self.K, H, W)

        # Every channels in x_unfold use the same weight. (G = 1)
        # Sum along neighbor axis and reshape.
        return (weight * x_unfold).sum(dim=3).view(B, self.input_channels, H, W)

if __name__ == "__main__":
    t = torch.rand((36, 128, 21, 21))
    i = Involution(128, 3, 16, 2, 1)
    print(i(t).size())
        
