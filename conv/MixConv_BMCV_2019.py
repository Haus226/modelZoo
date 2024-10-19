'''
Title
MixConv: Mixed Depthwise Convolutional Kernels

References
http://arxiv.org/abs/1907.09595
'''



import torch
import torch.nn as nn
import torch.nn.functional as F

class MixConv(nn.Module):
    def __init__(self, input_channels, kernels_size, bias=True):
        self.input_channels = input_channels
        self.kernels_size = kernels_size
        self.bias = bias
        self.convs = [nn.Conv2d(input_channels[idx], input_channels[idx], kernels_size[idx], bias=bias, 
                    groups=input_channels[idx], padding="same") 
                    for idx in range(len(kernels_size))]

    def forward(self, x):
        out = []
        x_split = torch.split(x, self.input_channels, dim=1)
        for idx in range(len(self.convs)):
            out.append(self.convs[idx](x_split[idx]))
            print(out[idx].size())
        return torch.cat(out, dim=1)


