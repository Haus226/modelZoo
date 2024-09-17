'''
Title
VanillaNet: the Power of Minimalism in Deep Learning

References
http://arxiv.org/abs/2305.12972
https://github.com/huawei-noah/VanillaNet/blob/main/models/vanillanet.py
'''



from torch import nn
import torch

class StackedAct(nn.Module):
    def __init__(self, channels, act_num=3, bias=True, deploy=False):
        super(StackedAct, self).__init__()
        self.channels = channels
        self.act_num = act_num
        self.bias = bias
        self.deploy = deploy
        self.block = nn.ModuleList([
            nn.Conv2d(channels, channels, act_num * 2 + 1, padding=act_num, groups=channels, bias=bias),
            nn.BatchNorm2d(channels) if not deploy else None
        ])
        # Another approach to replace the convolutional layer
        # The number of input channel is 1 since we are using group convolution.
        # self.weight = nn.Parameter(torch.rand((channels, 1, act_num * 2 + 1, act_num * 2 + 1)))
        # self.bias = nn.Parameter(torch.rand((channels))) if bias else None
        # if not self.deploy:
        #     self.bn = nn.BatchNorm2d(channels)

    def act(self, x):
        raise NotImplementedError

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
        x = self.act(x)
        for layer in self.block:
            x = layer(x)
        return x 