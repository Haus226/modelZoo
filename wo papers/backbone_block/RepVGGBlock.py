'''
Title
RepVGG: Making VGG-style ConvNets Great Again

References
http://arxiv.org/abs/2101.03697
https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
'''



import torch
from torch import nn
import torch.nn.functional as F

def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, groups=1):
    block = nn.Sequential()
    block.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias))
    block.add_module('bn', nn.BatchNorm2d(out_channels))
    return block

class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                stride=1, padding=0, dilation=1, groups=1, deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pad = [(kernel_size - 1) // 2] * 4

        self.identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
        self.conv = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.conv1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, groups=groups)
    
    def _fuse_conv_bn(self, branch):
        conv, bn = branch.conv, branch.bn
        gamma = bn.weight
        std = (bn.running_var + bn.eps).sqrt()
        weight = (gamma / std).view(-1, 1, 1, 1) * conv.weight.data
        bias = bn.bias + (conv.bias if conv.bias is not None else 0 - bn.running_mean) * (gamma / std)
        return weight, bias

    def _convert_idn(self):
        if not self.identity:
            return 0, 0
        in_channels = self.out_channels // self.groups
        # When identity branch is not None, in_channels == out_channels
        weight = torch.zeros((self.out_channels, in_channels, self.kernel_size, self.kernel_size)).float()
        for c in range(self.out_channels):
            weight[c, c % in_channels, 1, 1] = 1
        gamma = self.identity.weight
        std = (self.identity.running_var + self.identity.eps).sqrt()
        weight = (gamma / std).view(-1, 1, 1, 1) * weight
        bias = self.identity.bias - self.identity.running_mean * gamma / std
        return weight, bias

    def _get_fused_weight(self):
        weight_1, bias_1 = self._fuse_conv_bn(self.conv)
        weight_2, bias_2 = self._fuse_conv_bn(self.conv1x1)
        weight_3, bias_3 = self._convert_idn(self.identity)
        return weight_1 + F.pad(weight_2, self.pad) + weight_3,  + weight_3, bias_1 + bias_2 + bias_3

    def switch_to_deploy(self):
        if self.deploy:
            return
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, stride=self.conv.conv.stride,
                            padding=self.conv.conv.padding, dilation=self.conv.conv.dilation, groups=self.conv.conv.groups, bias=True)
        weight, bias = self._get_fused_weight()
        self.conv.weight.data = weight
        self.conv.bias.data = bias
        self.__delattr__("conv1x1")
        self.__delattr__('identity')
        self.deploy = True

    def forward(self, x):
        if self.deploy:
            return self.conv(x)
        else:
            idn = self.identity(x) if self.identity else 0
            return (self.conv(x) + self.conv1x1(x) + idn).relu()

if __name__ == "__main__":
    torch.manual_seed(226)
    t = torch.randn((32, 64, 32, 32))
    repvgg = RepVGGBlock(64, 16, 5, padding=2)
    print(repvgg.forward(t).size())