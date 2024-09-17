import torch
from torch import nn

'''
h_sigmoid and h_swish are obtained from https://github.com/houqb/CoordAttention/blob/main/coordatt.py
'''

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt:
    def __init__(self, input_channels, r):
        self.gap_h = nn.AdaptiveMaxPool2d((None, 1))
        self.gap_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(input_channels, input_channels // r, kernel_size=1) 
        self.bn = nn.BatchNorm2d(input_channels // r)
        self.act = h_swish()

        self.conv2 = nn.Conv2d(input_channels // r, input_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(input_channels // r, input_channels, kernel_size=1)

    def forward(self, x):
        n, b, h, w = x.size()
        pool_h= self.gap_h(x)
        # permute to concat
        pool_w = self.gap_w(x).permute(0, 1, 3, 2)
        c = torch.cat([pool_h, pool_w], dim=2)
        c = self.act(self.bn(self.conv1(c)))
        split_h, split_w = torch.split(c, [h, w], dim=2)
        split_h = self.conv2(split_h).sigmoid()
        split_w = self.conv3(split_w.permute(0, 1, 3, 2)).sigmoid()

        return x * split_h * split_w

if __name__ == '__main__':
    t = torch.rand(5, 9, 64, 64)
    m = CoordAtt(9, 3)
    print(m.forward(t).size())
