'''
Title
VanillaNet: the Power of Minimalism in Deep Learning

References
http://arxiv.org/abs/2305.12972
https://github.com/huawei-noah/VanillaNet/blob/main/models/vanillanet.py
'''



from torch import nn
import torch
from utils import ConvBNReLU

class StackedAct(nn.Module):
    def __init__(self, channels, act_num=3, bias=True, deploy=False):
        super(StackedAct, self).__init__()
        self.channels = channels
        self.act_num = act_num
        self.bias = bias
        self.deploy = deploy
        self.block = nn.ModuleList([
            nn.ReLU(),
            nn.Conv2d(channels, channels, act_num * 2 + 1, padding=act_num, groups=channels, bias=bias),
            nn.BatchNorm2d(channels) if not deploy else None
        ])
        # Another approach to replace the convolutional layer
        # The number of input channel is 1 since we are using group convolution.
        # self.weight = nn.Parameter(torch.rand((channels, 1, act_num * 2 + 1, act_num * 2 + 1)))
        # self.bias = nn.Parameter(torch.rand((channels))) if bias else None
        # if not self.deploy:
        #     self.bn = nn.BatchNorm2d(channels)

    def _fuse_conv_bn(self):
        _, conv, bn = self.block
        gamma = bn.weight
        std = (bn.running_var + bn.eps).sqrt()
        weight = (gamma / std).view(-1, 1, 1, 1) * conv.weight.data
        bias = bn.bias + (conv.bias - bn.running_mean) * (gamma / std)
        return weight, bias
    
    def switch_to_deploy(self):
        if self.block[1].bias is None:
            self.block[1].bias = nn.Parameter(torch.zeros(self.channels))
        weight, bias = self._fuse_conv_bn()
        self.block[2].weight.data = weight
        self.block[2].bias.data = bias
        del self.block[2]
        self.deploy = True
    
    def forward(self, x):
        for layer in self.block:
            x = layer(x)
        return x 
    
class VanillaBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_num=3, stride=2, bias=True, deploy=False):
        super(VanillaBlock, self).__init__()
        self.act_num = act_num
        self.lambda_ = 1
        self.deploy = deploy
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.conv_bn_1 = ConvBNReLU(in_channels, in_channels, 1, act=False)
        self.conv_bn_2 = ConvBNReLU(in_channels, out_channels, 1, act=False)
        self.act = StackedAct(out_channels, act_num, bias, deploy)
        self.pool = nn.Identity if stride == 1 else nn.MaxPool2d(stride)

    def _fuse_conv_bn(self, conv, bn):
        gamma = bn.weight
        std = (bn.running_var + bn.eps).sqrt()
        weight = (gamma / std).view(-1, 1, 1, 1) * conv.weight.data
        bias = bn.bias + (conv.bias - bn.running_mean) * (gamma / std)
        return weight, bias
    
    def switch_to_deploy(self):
        weight, bias = self._fuse_conv_bn(self.conv_bn_1.block.conv, self.conv_bn_1.block.bn)
        self.conv_bn_1.block.conv.weight.data = weight
        self.conv_bn_1.block.conv.bias.data = bias
        weight, bias = self._fuse_conv_bn(self.conv_bn_2.block.conv, self.conv_bn_2.block.bn)
        self.conv.weight.data = (weight.squeeze() @ self.conv_bn_1.block.conv.weight.data.squeeze()).view(weight.size())
        self.conv.bias.data = bias + (weight.squeeze() @ self.conv_bn_1.block.conv.bias.data.view(-1, 1)).squeeze()
        self.__delattr__('conv_bn_1')
        self.__delattr__('conv_bn_2')
        self.act.switch_to_deploy()
        self.deploy = True

    def forward(self, x):
        if self.deploy:
            x = self.conv(x)
        else:
            x = self.conv_bn_1(x)
            # Deep training strategy with ReLU
            x = torch.nn.functional.leaky_relu(x, self.lambda_)            
            x = self.conv_bn_2(x)
        x = self.pool(x)
        x = self.act(x)
        return x
    
if __name__ == "__main__":
    torch.manual_seed(226)
    t = torch.randn((32, 64, 32, 32))
    vanilla = VanillaBlock(64, 16)
    print(vanilla(t).size())
    vanilla.switch_to_deploy()
    print(vanilla(t).size())
