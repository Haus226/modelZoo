Concept

- Capture spatial correlations between channels by assigning weights to channels based on their significance. The weights are learned through a MLP layer
    
    - Squeeze
        
        - Squeeze  the global spatial information along the channel axis by global average pooling.
        - $z_c=F_sq(u_c)=\frac{1}{H\times W}\sum^{H}_{i=1}\sum^{W}_{j=1}u_c(i,j)$
    - Excitation
        
        - Must be capable of learning nonlinear interaction between channels
        - Learn a non-mutually exclusive relationship (Multiple channels are allowed to be emphasized
        - Simple gating mechanism by using a bottleneck MLP with two FC layers.
        - $s=F_ex(z, W)=\sigma(W_2\delta(W_1z))$
        - where $\sigma$ is sigmoid function, $\delta$ is ReLU function and $W_1\in\mathbb{R}^{C/r\times C},W_2\in\mathbb{R}^{C\times C/r}$
        - Finally, the output is scaled to perform channel-wise multiplication between the scalar $s_c$ and the feature map $u_c\in\mathbb{R}^{H\times W}$
            

```
class SEBlock:
    def __init__(self, input_channels:int, r:int):
        self.r = r
        self.fc1 = nn.Linear(input_channels, input_channels // r, bias=False)
        self.fc2 = nn.Linear(input_channels // r, input_channels, bias=False)
        # self.squeeze = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b, c, h, w = x.size()
        squeeze = torch.sum(x, (2, 3)) / (h * w)
        # With AdaptiveAvgPool2d -> self.squeeze(x) -> size = (b, c, 1, 1)
        # squeeze = self.squeeze(x).view(b, c)
        excitation = F.sigmoid(self.fc2(F.relu(self.fc1(squeeze)))).view(b, c, 1, 1)
        return x * excitation.expand_as(x)
```