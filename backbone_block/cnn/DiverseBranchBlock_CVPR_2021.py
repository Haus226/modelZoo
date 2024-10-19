'''
Title
Diverse Branch Block: Building a Convolution as an Inception-like Unit

References
http://arxiv.org/abs/2103.13425
https://github.com/DingXiaoH/DiverseBranchBlock
'''



import torch
from torch import nn
import torch.nn.functional as F
from utils import ConvBNReLU
from collections import OrderedDict

def TransI_fuse_bn(weight, bias, bn):
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    weight = (gamma / std).view(-1, 1, 1, 1) * weight.data
    bias = bn.bias + (bias if bias is not None else 0 - bn.running_mean) * (gamma / std)
    return weight, bias

def TransII_add_branch(weights, biases):
    return sum(weights), sum(biases)

def TransIII_1x1_kxk(weight_1x1, bias_1x1, weight_kxk, bias_kxk, groups):
    if groups == 1:
        weight = F.conv2d(weight_kxk, weight_1x1.permute(1, 0, 2, 3))      #
        bias = (weight_kxk * bias_1x1.reshape(1, -1, 1, 1)).sum((1, 2, 3))
    else:
        weights = []
        biases = []
        weight_1x1_T = weight_1x1.permute(1, 0, 2, 3)
        weight_1x1_width = weight_1x1.size(0) // groups
        weight_kxk_width = weight_kxk.size(0) // groups
        # Similar to case that groups = 1 but just convolve with the corresponding grouped channels
        for g in range(groups):
            weights.append(F.conv2d(weight_kxk[g * weight_kxk_width:(g + 1) * weight_kxk_width, :, :, :], 
                                    weight_1x1_T[:, g * weight_1x1_width:(g + 1) * weight_1x1_width, :, :]))
            biases.append((weight_kxk[g * weight_kxk_width:(g + 1) * weight_kxk_width, :, :, :] * bias_1x1[g * weight_1x1_width:(g + 1) * weight_1x1_width].reshape(1, -1, 1, 1)).sum((1, 2, 3)))
        weight, bias = TransIV_depth_cat(weights, biases)
    return weight, bias + bias_kxk

def TransIV_depth_cat(weights, biases):
    return torch.cat(weights), torch.cat(biases)

def TransV_avg(channels, kernel_size, groups):
    # For group convolution where C_in=6, G=3, C_out=6:
    # C1 F11 F21 ---> C1 * F11 + C2 * F21
    # C2 F12 F22 ---> C2 * F12 + C2 * F22

    # C1 F11 F21 ---> C1 * F11 + C2 * F21
    # C2 F12 F22 ---> C2 * F12 + C2 * F22

    # C1 F11 F21 ---> C1 * F11 + C2 * F21
    # C2 F12 F22 ---> C2 * F12 + C2 * F22

    # Since AvgPool is depthwise we should make something like
    # C1 (F11 + Avg) (F21 + 0) 
    # C2 (F12 + 0) (F21 + Avg)
    # 0 is something like masked the certain channels (not performing AvgPool) 
    # so we can averaging the channels in depthwise while also 
    # perform the original convolution
    # The addition is due to that we actually sum the output features of each branches.
    in_channels = channels // groups
    weight = torch.zeros((channels, in_channels, kernel_size, kernel_size))
    weight[torch.arange(channels), torch.tile(torch.arange(in_channels), [groups]), :, :] = 1 / kernel_size ** 2
    # Another approach:
    # for c in range(channels):
        # weight[c, c % in_channels, :, :] = 1 / kernel_size ** 2
    return weight

# This has not been tested with non-square kernels (kernel.size(2) != kernel.size(3)) nor even-size kernels
def TransVI_pad_multi_scale(weight, target_size):
    h_pad = (target_size - weight.size(2)) // 2
    w_pad = (target_size - weight.size(3)) // 2
    return F.pad(weight, [h_pad, h_pad, w_pad, w_pad])

class IdentityConv1x1(nn.Conv2d):
    def __init__(self, channels, groups=1):
        super(IdentityConv1x1, self).__init__(channels, channels, 1, groups=groups, bias=False)

        in_channels = channels // groups
        self.idn = torch.zeros((channels, in_channels, 1, 1))
        for c in range(channels):
            self.idn[c, c % in_channels, 0, 0] = 1
        # Another approach:
        # self.idn[torch.arange(channels), torch.tile(torch.arange(in_channels), [groups]), 0, 0] = 1

        nn.init.zeros_(self.weight)
    
    def forward(self, x):
        return F.conv2d(x, self.weight + self.idn.to(self.weight.device), None, groups=self.groups)

    def get_weight(self):
        return self.weight + self.idn.to(self.weight.device)

class PaddedBN(nn.BatchNorm2d):
    def __init__(self, padding, channels, eps=1e-5, momentum=0.1, affine=True):
        super(PaddedBN, self).__init__(channels, eps, momentum, affine)
        self.padding = padding

    def forward(self, x):
        x = super(PaddedBN, self).forward(x)
        if self.padding:
            pad = self.bias if self.affine else 0 - self.running_mean * self.weight / torch.sqrt(self.running_var + self.eps)
            pad = pad.view(1, -1, 1, 1)
            x = F.pad(x, [self.padding] * 4)   
            # Set the padded zeros to values computed
            # Top, Bottom, Left, Right
            x[:, :, 0:self.padding, :] = pad
            x[:, :, -self.padding:, :] = pad
            x[:, :, :, 0:self.padding] = pad
            x[:, :, :, -self.padding:] = pad
        return x
    
class DiverseBranchBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                stride=1, padding=0, dilation=1, groups=1,
                inter_channels=None, deploy=False,
                act=None):
        super(DiverseBranchBlock, self).__init__()
        self.act = act if act else nn.Identity()
        self.kernel_size = kernel_size
        self.groups = groups
        self.out_channels = out_channels

        if deploy:
            self.conv_rep = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)

        self.conv = ConvBNReLU(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False, act=False)
        # Refer to Section 3.3
        self.conv_1x1 = ConvBNReLU(in_channels, out_channels, 1, stride, 0, dilation, groups, False, act=False) if groups < in_channels else None
        self.conv_avg = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(in_channels, out_channels, 1, 1, 0, groups=groups, bias=False) if groups < in_channels else None),
            ("bn", PaddedBN(padding, out_channels)  if groups < in_channels else None),
            ("avg", nn.AvgPool2d(kernel_size, stride, padding=0)),
            ("avg_bn", nn.BatchNorm2d(out_channels))
            ])
        )
        if inter_channels is None:
            inter_channels == in_channels if groups < in_channels else 2 * in_channels # For MobileNet
        self.conv_1x1_kxk = nn.Sequential(OrderedDict([
            ("idconv1x1", IdentityConv1x1(in_channels, groups)) if inter_channels == in_channels else
            ("conv1", nn.Conv2d(in_channels, inter_channels, 1, groups=groups, bias=False)),
            ("bn1", PaddedBN(padding, inter_channels)),
            ("conv2", nn.Conv2d(inter_channels, out_channels, kernel_size, stride, 0, groups=groups, bias=False)),
            ("bn2", nn.BatchNorm2d(out_channels))
            ])
        )

    def _get_fused_weight(self):
        weight_conv, bias_conv = TransI_fuse_bn(self.conv.conv.weight, self.conv.conv.bias, self.conv.bn)
        if self.conv_1x1 is not None:
            weight_1x1, bias_1x1 = TransI_fuse_bn(self.conv_1x1.conv.weight, self.conv_1x1.conv.bias, self.conv_1x1.bn)
            weight_1x1 = TransVI_pad_multi_scale(weight_1x1, self.kernel_size)
        else:
            weight_1x1, bias_1x1 = 0, 0

        if hasattr(self.conv_1x1_kxk, "idconv1x1"):
            weight_1x1_kxk_first = self.conv_1x1_kxk.idconv1x1.get_weight()
        else:
            weight_1x1_kxk_first = self.conv_1x1_kxk.conv1.weight
        weight_1x1_kxk_first, bias_1x1_kxk_first = TransI_fuse_bn(weight_1x1_kxk_first, 
                                                                self.conv_1x1_kxk.conv1.bias,
                                                                self.conv_1x1_kxk.bn1)
        weight_1x1_kxk_second, bias_1x1_kxk_second = TransI_fuse_bn(self.conv_1x1_kxk.conv2.weight, 
                                                                    self.conv_1x1_kxk.conv2.bias, 
                                                                    self.conv_1x1_kxk.bn2)
        weight_1x1_kxk_merged, bias_1x1_kxk_merged = TransIII_1x1_kxk(
            weight_1x1_kxk_first, bias_1x1_kxk_first,
            weight_1x1_kxk_second, bias_1x1_kxk_second,
            self.groups
        )

        weight_avg = TransV_avg(self.out_channels, self.kernel_size, self.groups)
        weight_1x1_avg_second, bias_1x1_avg_second = TransI_fuse_bn(weight_avg.to(self.conv_avg.avg_bn.weight.device), 
                                                                    None,
                                                                    self.conv_avg.avg_bn)
        if self.conv_avg.conv:
            weight_1x1_avg_first, bias_1x1_avg_first = TransI_fuse_bn(self.conv_avg.conv.weight, self.conv_avg.conv.bias, self.conv_avg.bn)
            weight_1x1_avg_merged, bias_1x1_avg_merged = TransIII_1x1_kxk(weight_1x1_avg_first, bias_1x1_avg_first,
                                                                        weight_1x1_avg_second, bias_1x1_avg_second,
                                                                        self.groups)
        else:
            weight_1x1_avg_merged, bias_1x1_avg_merged = weight_1x1_kxk_second, bias_1x1_avg_second
        
        return TransII_add_branch([weight_conv, weight_1x1, weight_1x1_kxk_merged, weight_1x1_avg_merged],
                                [bias_conv, bias_1x1, bias_1x1_kxk_merged, bias_1x1_avg_merged])
    
    def switch_to_deploy(self):
        if hasattr(self, "conv_rep"):
            return
        weight, bias = self._get_fused_weight()
        self.conv_rep = nn.Conv2d(
            self.conv.conv.in_channels, self.conv.conv.out_channels, 
            self.conv.conv.kernel_size, self.conv.conv.stride,
            self.conv.conv.padding, self.conv.conv.dilation, self.conv.conv.groups, bias=True
        )
        self.conv_rep.weight.data = weight
        self.conv_rep.bias.data = bias
        for p in self.parameters():
            p.detach_()
        self.__delattr__('conv')
        self.__delattr__('conv_avg')
        if hasattr(self, 'conv_1x1'):
            self.__delattr__('conv_1x1')
        self.__delattr__('conv_1x1_kxk')

    def forward(self, x):
        if hasattr(self, "conv_rep"):
            return self.act(self.conv_rep(x))
        
        y = self.conv(x)
        if hasattr(self, "conv_1x1"):
            y += self.conv_1x1(x)
        y += self.conv_avg(x)
        y += self.conv_1x1_kxk(x)
        return self.act(y)
    
if __name__ == "__main__":
    torch.manual_seed(226)
    t = torch.rand((32, 64, 21, 21))
    dbb = DiverseBranchBlock(64, 128, 3, padding=1, groups=8, inter_channels=96, act=nn.ReLU())
    branches = dbb(t)
    dbb.switch_to_deploy()
    merged = dbb(t)
    print(torch.allclose(branches, merged))
    print(torch.abs(branches - merged))
