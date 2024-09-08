import torch
from torch import nn
import torch.nn.functional as F

class ODConv:
    def __init__(self, input_channels, output_channels, kernel_size, kernel_num=4, r=16, groups=1, 
                stride=1, padding=0, dilation=1):
        inter_channels = input_channels // r

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_channels = output_channels
        self.groups = groups
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0
        self.weight = nn.Parameter(torch.rand(kernel_num, output_channels, input_channels // groups, kernel_size, kernel_size))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(input_channels, inter_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(inter_channels, input_channels, 1)
        self.spatial_fc = nn.Conv2d(inter_channels, kernel_size * kernel_size, 1)
        self.kernel_fc = nn.Conv2d(inter_channels, kernel_num, 1)
        self.filter_fc = nn.Conv2d(inter_channels, output_channels, 1)

    def get_attention(self, x):
        B, _, _, _ = x.size()
        x = self.gap(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)

        # From author:
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_att = (self.channel_fc(x).view(B, -1, 1, 1) / self.temperature).sigmoid()
        filter_att = (self.filter_fc(x).view(B, -1, 1, 1) / self.temperature).sigmoid()

        # Multiplying with weight
        spatial_att = (self.spatial_fc(x).view(B, 1, 1, 1, self.kernel_size, self.kernel_size) / self.temperature).sigmoid()
        kernel_att = (self.kernel_fc(x).view(B, -1, 1, 1, 1, 1) / self.temperature).softmax(dim=1)
        return channel_att, filter_att, spatial_att, kernel_att
    
    def forward(self, x):
        B, C, H, W = x.size()
        channel_att, filter_att, spatial_att, kernel_att = self.get_attention(x)
        x = x * channel_att
        x = x.view(1, B * C, H, W)
        self.weight = spatial_att * kernel_att * self.weight

        # Sum along the kernel_num dimension
        self.weight = self.weight.sum(dim=1).view(-1, C // self.groups, self.kernel_size, self.kernel_size)
        y = F.conv2d(x, weight=self.weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * B)
        y = y.view(B, self.output_channels, y.size(2), y.size(3))
        return y * filter_att

if __name__ == "__main__":
    t = torch.rand(32, 64, 21, 21)
    odconv = ODConv(64, 16, 3, padding=1)
    print(odconv.forward(t).size())