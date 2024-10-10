'''
Title
Pyramidal Convolution: Rethinking Convolutional Neural Networks for Visual Recognition

References
http://arxiv.org/abs/2006.11538
'''



import torch
from torch import nn

class PyConv(nn.Module):
    def __init__(self, input_channels, output_channels, kernels_size, groups, stride=1, dilation=1, bias=True):
        assert len(output_channels) == len(kernels_size) == len(groups)
        super(PyConv, self).__init__()

        self.levels = []
        for idx in range(len(kernels_size)):
            self.levels.append(nn.Conv2d(
                input_channels, output_channels[idx], kernels_size[idx],
                stride=stride, dilation=dilation, bias=bias, padding="same"
            ))
        self.levels = nn.ModuleList(self.levels)

    def forward(self, x):
        outputs = []
        for c in self.levels:
            outputs.append(c.forward(x))
        return torch.cat(outputs, dim=1)
    
if __name__ == "__main__":
    torch.manual_seed(42)
    t = torch.rand(5, 12, 64, 64).float()
    s = PyConv(12, [4, 5, 3], [3, 5, 7], [2, 1, 4])
    print(s(t).size())
                