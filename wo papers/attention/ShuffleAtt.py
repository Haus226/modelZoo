import torch
from torch import nn


class ShuffleAtt:
    def __init__(self, groups, input_channels):
        self.groups = groups
        channels = input_channels // (2 * self.groups)

        # The input channel: C_in // 2G is divided into C_in // 2G groups --> each containing 1 channel.
        self.gn = nn.GroupNorm(channels, channels)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.channel_weight = nn.Parameter(torch.rand(1, channels, 1, 1))
        self.channel_bias = nn.Parameter(torch.rand(1, channels, 1, 1))
        self.spatial_weight = nn.Parameter(torch.rand(1, channels, 1, 1))
        self.spatial_bias = nn.Parameter(torch.rand(1, channels, 1, 1))
        
        # Another approach: For learning channel-wise weight and bias (the channels do not affect one another)
        # self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, groups=channels)
        # self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, groups=channels)
        # self.conv1.weight.data = self.channel_weight.view(channels, 1, 1, 1).clone()
        # self.conv1.bias.data = self.channel_bias.view(channels).clone()
        # self.conv2.weight.data = self.spatial_weight.view(channels, 1, 1, 1).clone()
        # self.conv2.bias.data = self.spatial_bias.view(channels).clone()


    def forward(self, x):
        b, c, h, w = x.size()
        x_sub = x.view(b * self.groups, c // self.groups, h, w)
        x_c, x_s = torch.chunk(x_sub, 2, dim=1)

        channel_part = self.gap(x_c)
        # Broadcasting
        # channel_part_ = self.conv1(channel_part)
        channel_part = self.channel_weight * channel_part + self.channel_bias
        # print((channel_part - channel_part_).sum())
        channel_part = channel_part.sigmoid()
        x_c = x_c * channel_part

        spatial_part = self.gn(x_s)
        # spatial_part_ = self.conv2(spatial_part)
        spatial_part = self.spatial_weight * spatial_part + self.spatial_bias
        # print((spatial_part - spatial_part_).sum())
        spatial_part = spatial_part.sigmoid()
        x_s = x_s * spatial_part

        x_cat = torch.cat([x_c, x_s], dim=1).view(b, c, h, w)

        # Channel shuffle
        return x_cat.view(b, 2, c // 2, h, w).permute(0, 2, 1, 3, 4).reshape(b, c, h, w)

if __name__ == "__main__":
    t = torch.rand(5, 12, 64, 64)
    m = ShuffleAtt(2, 12)
    print(m.forward(t).size())
