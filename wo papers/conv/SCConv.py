import torch
from torch import nn

class SRU:
    def __init__(self, input_channels, groups=16, threshold=0.5) -> None:
        self.gn = nn.GroupNorm(groups, input_channels)
        self.threshold = threshold

    def forward(self, x):
        B, C, H, W = x.size()
        norm_x = self.gn.forward(x)
        norm_gamma = self.gn.weight / self.gn.weight.sum()
        w = (norm_x * norm_gamma.view((1, C, 1, 1))).sigmoid()
        # Gating mechanism to get the locations of informative and non-informative spatial
        w1, w2 = w >= self.threshold, w < self.threshold

        # Filtering the inputs to informative and non-informative spatial features
        x_w1, x_w2 = x * w1, x * w2
        x_w11, x_w12 = torch.chunk(x_w1, 2, dim=1)
        x_w21, x_w22 = torch.chunk(x_w2, 2, dim=1)
        return torch.cat([x_w11 + x_w22, x_w21 + x_w12], dim=1)

class CRU:
    def __init__(self, input_channels, kernel_size=3, alpha=0.5, r=2, groups=2) -> None:
        self.up_channel = int(alpha * input_channels)
        self.low_channel = input_channels - self.up_channel

        self.squeeze1 = nn.Conv2d(self.up_channel, self.up_channel // r, 1, bias=False)
        self.squeeze2 = nn.Conv2d(self.low_channel, self.low_channel // r, 1, bias=False)

        self.GWC = nn.Conv2d(self.up_channel // r, input_channels, kernel_size, 
                             padding=kernel_size // 2, groups=groups)
        self.PWC1 = nn.Conv2d(self.up_channel // r, input_channels, 1, bias=False)
        self.PWC2 = nn.Conv2d(self.low_channel // r, input_channels - self.low_channel // r, 1, bias=False)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(x1), self.squeeze2(x2)
        y1 = self.GWC(up) + self.PWC1(up)
        y2 = torch.cat([self.PWC2(low), low], dim=1)
        y = torch.cat([y1, y2], dim=1)
        y = y * self.gap(y).softmax(dim=1)
        y1, y2 = torch.chunk(y, 2, dim=1)
        return y1 + y2
        
class SCConv:
    def __init__(self, input_channels, threshold=0.5, alpha=0.5, r=2, gn_groups=16, gwc_groups=2,
                gwc_kernel=3):
        self.sru = SRU(input_channels, gn_groups, threshold)
        self.cru = CRU(input_channels, gwc_kernel, alpha, r, gwc_groups)

    def forward(self, x):
        x = self.sru.forward(x)
        x = self.cru.forward(x)
        return x

if __name__ == "__main__":
    t = torch.rand(32, 64, 21, 21)
    sc = SCConv(64)
    sc.forward(t).size()