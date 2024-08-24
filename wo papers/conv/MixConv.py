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
if __name__ == "__main__":
    t = torch.rand(5, 10, 64, 64)
    m = MixConv([2, 2, 3, 3], [3, 5, 7, 9])
    m.forward(t).size()