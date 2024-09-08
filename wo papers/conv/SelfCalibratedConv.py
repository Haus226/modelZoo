import torch.nn.functional as F
import torch
from torch import nn

class SelfCalibratedConv:
    def __init__(self, input_channels, output_channels, 
                kernel_size, 
                stride, padding, dilation=1, groups=1,
                pooling_r=2):
        self.idn = (input_channels == output_channels)
        in_channels = in_channels
        out_channels = out_channels
        if not self.idn:
            self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.k1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.k3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, dilation, groups, bias=False),
            nn.BatchNorm2d(out_channels),        
        )
        self.k4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.convBlock1 = nn.Sequential(
            nn.Conv2d(input_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        self.convBlock2 = nn.Sequential(
            nn.Conv2d(input_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

    def scconv(self, x):
        identity = x if self.idn else self.pwc(x)
        y = torch.sigmoid(identity + F.interpolate(self.k2(x), identity.size()[2:]))
        y = self.k3(x) * y
        return  self.k4(y)

    def forward(self, x):
        x1, x2 = self.convBlock1(x), self.convBlock2(x)
        x1 = self.k1(x1).relu()
        x2 = self.scconv(x2).relu()
        return torch.cat([x1, x2], dim=1)

if __name__ == "__main__":
    t = torch.rand(32, 64, 21, 21)
    scconv = SelfCalibratedConv(64, 16, 3, 1, 1, 1, 1)
    print(scconv.forward(t).size())
