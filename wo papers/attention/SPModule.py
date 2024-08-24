import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, bias, relu):
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, bias)
        self.bn = nn.BatchNorm2d(output_channels)
        self.is_relu = relu
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return nn.ReLU()(x) if self.is_relu else x

class SPM:
    def __init__(self, 
                input_channels:int, 
                output_channels:int,
                ):
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)
        # Another approach:
        # self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=(3, 1), padding=(1, 0), bias=False)
        # self.conv2 = nn.Conv2d(input_channels, output_channels, kernel_size=(1, 3), padding=(0, 1), bias=False)
        # self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        # self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        self.conv3 = nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=True)

    def forward(self, x):
        vertical = torch.mean(x, dim=3)
        horizontal = torch.mean(x, dim=2)
        vertical = self.conv1(vertical).unsqueeze(3)
        horizontal = self.conv2(horizontal).unsqueeze(2) 

        # Another appraoch:
        # vertical = self.pool1(x)
        # horizontal = self.pool2(x)
        # vertical = self.conv1(vertical)
        # horizontal = self.conv2(horizontal)

        vertical = self.bn1(vertical)
        horizontal = self.bn2(horizontal)

        fusion = horizontal + vertical # Broadcast
        fusion = self.conv3(F.relu(fusion))
        return fusion.sigmoid()

class MPM:
    def __init__(self, input_channels, r:int, pool_size, **up_kwargs):
        self.up_kwargs = up_kwargs
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = input_channels // r
        self.reduce1 = ConvBlock(input_channels, inter_channels, 1, 1, 1, False, True)
        self.reduce2 = ConvBlock(input_channels, inter_channels, 1, 1, 1, False, True)

        # For short range dependencies
        self.s_conv1 = ConvBlock(inter_channels, inter_channels, 3, 1, 1, False, False)
        self.s_conv2 = ConvBlock(inter_channels, inter_channels, 3, 1, 1, False, False)
        self.s_conv3 = ConvBlock(inter_channels, inter_channels, 3, 1, 1, False, False)

        # For long range dependencies (SPM)
        self.l_conv1 = ConvBlock(inter_channels, inter_channels, (1, 3), 1, (0, 1), False, False)
        self.l_conv2 = ConvBlock(inter_channels, inter_channels, (3, 1), 1, (1, 0), False, False)

        self.conv1 = ConvBlock(inter_channels, inter_channels, 3, 1, 1, False, True)
        self.conv2 = ConvBlock(inter_channels, inter_channels, 3, 1, 1, False, True)
        self.conv3 = ConvBlock(inter_channels * 2, input_channels, 1, 1, 1, False, False)
        
    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.reduce1(x)
        x2 = self.reduce2(x)

        x1_1 = self.s_conv1(x1)
        x1_2 = F.interpolate(self.s_conv2(self.pool1(x1)), (h, w), **self._up_kwargs)
        x1_3 = F.interpolate(self.s_conv3(self.pool2(x1)), (h, w), **self._up_kwargs)

        x2_1 = F.interpolate(self.l_conv1(self.pool3(x2)), (h, w), **self._up_kwargs)
        x2_2 = F.interpolate(self.l_conv2(self.pool4(x2)), (h, w), **self._up_kwargs)

        x1 = self.conv1(F.relu_(x1_1 + x1_2 + x1_3))
        x2 = self.conv2(F.relu_(x2_1 + x2_2))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)
        

