'''
Title
CondConv: Conditionally Parameterized Convolutions for Efficient Inference

References
http://arxiv.org/abs/1904.04971
'''



import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair

class CondConv(_ConvNd):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1,
                bias=True, padding_mode='zeros', num_experts=3):

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        super(CondConv, self).__init__(
                    self.input_channels, self.output_channels, self.kernel_size, 
                    self.stride, self.padding, self.dilation,
                    False, _pair(0), groups, bias, padding_mode)

        self.num_experts = num_experts
        self.weight_shape = (self.output_channels, self.input_channels, kernel_size)
        num_param = self.output_channels * (self.input_channels // self.groups) * kernel_size * kernel_size
        self.weight = nn.Parameter(torch.Tensor(self.num_experts, num_param))
        self.bias = None
        if bias:
            self.bias_shape = (self.output_channels, )
            self.bias = nn.Parameter(torch.Tensor(self.num_experts, self.output_channels))
        self.routing = nn.Linear(input_channels, num_experts)

    def routingBlock(self, x):
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)  # (B, C, H, W) --> (B, C, 1, 1) --> (B, C)
        # (B, C) @ (C, M) --> (B, M)
        # A matrix where its entries are the weight of j_th kernel for the i_th batch
        return torch.sigmoid(self.routing(pooled_inputs))

    def forward(self, x):
        B, C, H, W = x.size()
        routing_weights = self.routingBlock(x)
        # (B, M) @ (M, OC * IC / G * K * K) --> (B, OC * IC / G * K * K)
        # Ex:
        # [a, b, c]     [k1]              [a * k1 + b * k2 + c * k3]
        # [d, e, f]     [k2]      --->    [d * k1 + e * k2 + f * k3]
        # [g, h, i]     [k3]              [g * k1 + h * k2 + i * k3]
        weight = torch.matmul(routing_weights, self.weight)
        # (B * OC, IC / G, K, K)
        weight = weight.view(B * self.output_channels, self.input_channels // self.groups, *self.kernel_size) 
        bias = None
        if self.bias is not None:
            bias = torch.matmul(routing_weights, self.bias)
            bias = bias.view(B * self.output_channels)
        # move batch elements with channels so each batch element can be efficiently convolved with separate kernel
        # reshape instead of view to work with channels_last input and using for loop
        x = x.view(1, B * C, H, W)
        if self.padding_mode == "zeros":
            out = F.conv2d(
                x, weight, bias, stride=self.stride, padding="valid",
                dilation=self.dilation, groups=self.groups * B)
        else:
            out = F.conv2d(F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, stride=self.stride,
                            dilation=self.dilation, groups=self.groups * B)
        return out.view(B, self.output_channels, *out.size()[-2:])
    
if __name__ == "__main__":
    t = torch.rand(5, 64, 21, 21)
    s = CondConv(64, 128, 3)
    print(s(t).size())