from torch import nn
import torch

class SGE:
    def __init__(self, groups, input_channels):
        self.groups = groups
        self.gap = nn.AdaptiveAvgPool2d(1)

        # weight and bias for each groups
        self.weight   = nn.Parameter(torch.ones(1, self.groups, 1, 1))
        self.bias     = nn.Parameter(torch.zeros(1, self.groups, 1, 1))
        self.sig      = nn.Sigmoid()

        # Another approach: For learning group-wise weight and bias (the groups do not affect one another)
        # self.conv = nn.Conv2d(self.groups, self.groups, 1, groups=self.groups)
        # self.conv.weight.data = self.weight.view(self.groups, 1, 1, 1).clone()
        # self.conv.bias.data = self.bias.view(self.groups).clone()

        self.gn = nn.GroupNorm(self.groups, input_channels)


    def forward(self, x):
        x_pool = x * self.gap(x)
        t = self.gn.forward(x_pool).sigmoid()
        return x * t

    def oforward(self, x):
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w) 
        xn = x * self.gap(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x

if __name__ == "__main__":
    torch.manual_seed(42)
    t = torch.rand(5, 1024, 64, 64)
    m = SGE(64, 1024)
    print(m.forward(t).size())
    print(m.oforward(t).size())