Concept

- Slightly different from CondConv ([Yang et al., 2020]
    
    - CondConv use a Average Pooling followed by a linear layer with sigmoid as activation function.
    - DynamicConv use SE block as attention with sigmoid replaced by softmax. The softmax is applied along the num_experts axis after the last linear layer.
    - The paper argues that softmax helps the learning of attention block.
    - However, softmax does NOT work well on this due to its near one-hot output. It only allows a small subset of kernels across layers to be optimized.
    - Therefore the paper suggests to introduce a temperature parameter in the softmax function to lead to near-uniform attention across kernels during early training. This enables simultaneous optimization of multiple kernels, resulting in faster convergence.
    - Temperature annealing further enhances performance.

```
class DynamicConv(_ConvNd):
    def __init__(self, input_channels, output_channels, kernel_size, r, temp, stride=1,
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
        super(DynamicConv, self).__init__(
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
        self.r = r
        self.temperature = temp
        self.fc1 = nn.Linear(input_channels, input_channels // r, bias=False)
        self.fc2 = nn.Linear(input_channels // r, self.num_experts, bias=False)
        self.squeeze = nn.AdaptiveAvgPool2d(1)

    def routingBlock(self, x):
        b, c, h, w = x.size()
        squeeze = self.squeeze(x).view(b, c)
        t = self.fc2(F.relu(self.fc1(squeeze))).view(b, self.num_experts)
        excitation = torch.softmax(t / self.temperature, dim=-1)
        self.updateTemperature()
        return excitation
    
    def updateTemperatue(self):
        raise NotImplementedError

    def forward(self, x):
        B, C, H, W = x.size()
        routing_weights = self.routingBlock(x)
        weight = torch.matmul(routing_weights, self.weight)
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
```