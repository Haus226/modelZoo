Concept

- Suppose a convolution layer has $I$ input channels, $O$ output channels with kernel size of $k$ then, CondConv is just equivalent of $N$ of such layers and passing the input feature to these layers in parallel.
    
    - Shape of tensor in Conv: $[I, O, k, k]$
    - Shape of tensor in CondConv:$[N, I, O, k, k]$
- Instead of passing the input in parallel and learn the weights between the $N$ layers, the paper shows a equivalent but cheaper way by distributive property of convolution.
    
    - $F(x)=\alpha_1(W_1\ast x)+\cdots+\alpha_n(W_n\ast x)=(\alpha_1W_1)\ast x+\cdots+(\alpha_nW_n)\ast x=(\alpha_1W_1+\cdots\alpha_nW_n)\ast x$

```
class CondConv2D(_ConvNd):
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
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)  # CondConv routing
        return torch.sigmoid(self.routing(pooled_inputs))

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