from torch import nn
import torch

class scSE:
    def __init__(self, input_channels):
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, input_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(input_channels // 2, input_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.sse = nn.Sequential(
            nn.Conv2d(input_channels, 1, kernel_size=1, padding="same"),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.cse(x) + x * self.sse(x)

if __name__ == "__main__":
    t = torch.rand(5, 1024, 64, 64)
    m = scSE(1024)
    print(m.forward(t).size())