import torch
import torch.nn as nn
import torch.nn.functional as F

class SRM:
    def __init__(self, input_channels):
        # groups = input_channels --> depthwise/channelwise
        # kernel_size = spatial_size --> fully connected layer
        # output_size = [(I - K + 2P) / S] + 1 = 1
        self.cfc = nn.Conv1d(input_channels, input_channels, kernel_size=2, bias=False,
                            groups=input_channels)
        self.bn = nn.BatchNorm1d(input_channels)

    def forward(self, x):
        b, c, _, _ = x.size()

        # Style pooling
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat((mean, std), -1)  # (b, c, 2)

        # Style integration
        z = self.cfc(u)  # (b, c, 1)
        z = self.bn(z)
        g = torch.sigmoid(z)
        g = g.view(b, c, 1, 1)

        return x * g       
if __name__ == "__main__":
    t = torch.arange(405).view(5, 9, 3, 3).float()
    s = SRM(9)
    s.forward(t).size()