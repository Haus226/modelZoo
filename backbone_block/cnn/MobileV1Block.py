import torch
from torch import nn
from utils import ConvBNReLU

class MobileV1Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                expansion=4, alpha=1.0):
        super(MobileV1Block, self).__init__()
        channels = int(alpha * in_channels * expansion)
        self.conv = ConvBNReLU(int(alpha * in_channels), channels, 1)
        self.dw_conv = ConvBNReLU(channels, channels, kernel_size, stride, padding, dilation, channels)
        self.pw_conv = ConvBNReLU(channels, int(alpha * out_channels), 1, stride)

    def forward(self, x):
        return self.pw_conv(self.dw_conv(self.conv(x)))
    
if __name__ == "__main__":
    torch.manual_seed(226)
    alpha = 0.75
    t = torch.rand((32, int(64 * alpha), 21, 21))
    mobile = MobileV1Block(64, 256, 3, alpha=alpha)
    print(mobile(t).size())

