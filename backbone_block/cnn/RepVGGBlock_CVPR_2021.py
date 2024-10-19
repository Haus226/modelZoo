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
from utils import ConvBNReLU

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
        self.conv = ConvBNReLU(in_channels, out_channels, kernel_size, stride, padding, groups, act=False)
        self.conv1x1 = ConvBNReLU(in_channels, out_channels, 1, stride, groups=groups, act=False)
    
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
        # When identity branch is not None, in_channels == out_channels
        in_channels = self.out_channels // self.groups
        weight = torch.zeros((self.out_channels, in_channels, self.kernel_size, self.kernel_size)).float()
        # Try to draw an example for better understanding, this is basically try to convert
        # identity branch to an equivalent shape with the other branch for merging.
        # When groups = 1, suppose there is C_in input channels and D_out = C_in output channels (Remember this is identity mapping)
        # C1   F1 F2 ... F_in D1
        # C2                  D2
        # ...                 ...  
        # C_in                D_out
        # So, the possible way is performing depthwise convolution where the corresponding filter 
        # having same kernel size as the original branch but with the center equals to 1 for identity mapping.
        # When groups = G, suppose there is 6 input(output) channels and G = 2, where F is now the
        # weight of the original convolution we wish to merge.
        # C1 F11 F21 ---> C1 * F11 + C2 * F21
        # C2 F12 F22 ---> C2 * F12 + C2 * F22

        # C1 F11 F21 ---> C1 * F11 + C2 * F21
        # C2 F12 F22 ---> C2 * F12 + C2 * F22

        # C1 F11 F21 ---> C1 * F11 + C2 * F21
        # C2 F12 F22 ---> C2 * F12 + C2 * F22
        # Since the identity mapping should be depthwise, we should make something like
        # C1 (F11 + Idn) (F21 + 0) 
        # C2 (F12 + 0) (F21 + Idn)
        # 0 is something like masked the certain channels (not performing Identity) 
        # so we can map the channels in depthwise while also 
        # perform the original convolution
        # The addition is due to that we actually sum the output features of each branches.

        for c in range(self.out_channels):
            weight[c, c % in_channels, self.kernel_size // 2, self.kernel_size // 2] = 1
        gamma = self.identity.weight
        std = (self.identity.running_var + self.identity.eps).sqrt()
        weight = (gamma / std).view(-1, 1, 1, 1) * weight
        bias = self.identity.bias - self.identity.running_mean * gamma / std
        return weight, bias

    def _get_fused_weight(self):
        weight_1, bias_1 = self._fuse_conv_bn(self.conv)
        weight_2, bias_2 = self._fuse_conv_bn(self.conv1x1)
        weight_3, bias_3 = self._convert_idn(self.identity)
        return weight_1 + F.pad(weight_2, self.pad) + weight_3, bias_1 + bias_2 + bias_3

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
    print(repvgg(t).size())