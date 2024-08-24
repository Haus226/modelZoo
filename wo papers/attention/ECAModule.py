import torch
from torch import nn

class ECA:
    def __init__(self, kernel_size, bias=False):
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding="same", bias=bias)
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).squeeze(-1)
        # From (N, C, 1) -> (N, 1, C) so that convolution slides along channel dimension
        y = self.conv(y.transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * y.sigmoid().view(b, c, 1, 1)
    
if __name__ == '__main__':
    t = torch.rand(5, 9, 64, 64)
    m = ECA(3)
    print(m.forward(t).size())
