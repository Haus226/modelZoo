import torch
from torch import nn
from torch.nn import functional as F

class SEBlock:
    def __init__(self, input_channels:int, r:int):
        self.r = r
        self.fc1 = nn.Linear(input_channels, input_channels // r, bias=False)
        self.fc2 = nn.Linear(input_channels // r, input_channels, bias=False)
        # self.squeeze = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b, c, h, w = x.size()
        squeeze = torch.sum(x, (2, 3)) / (h * w)
        # With AdaptiveAvgPool2d -> self.squeeze(x) -> size = (b, c, 1, 1)
        # squeeze = self.squeeze(x).view(b, c)
        excitation = F.sigmoid(self.fc2(F.relu(self.fc1(squeeze)))).view(b, c, 1, 1)
        return x * excitation.expand_as(x)