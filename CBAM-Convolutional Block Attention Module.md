Concept

- Sequentially infers a $1D$ channel attention map $M_c\in R^{C\times1\times1}$ and a $2D$ spatial attention map $M_s\in R^{1\times H\times W}$
    
    - Channel Attention
        
        - Pass input feature to MaxPool and AvgPool in parallel.
        - Both outputs are then forwarded to a shared MLP network with one hidden layer. The hidden activation size is $C/r$ where $C$ is the input feature of the first linear layer of MLP network and a hyperparameter $r$, the reduction ratio of neurons.
            
            - The paper fixes $r=16$  and did not mention any experiments conduct to validate this value.
        - The outputs are merged using element-wise summation.
        - $M_c(F)=\sigma(MLP(AvgPool(F))+MLP(MaxPool(F)))$
        - $MLP=W_1(Relu(W_0(\cdot)))$
        - where $\sigma$ is sigmoid function and $W_0\in\mathbb{R}^{C/r\times C},W_1\in\mathbb{R}^{C\times C/r}$
    - Spatial Attention
        
        - Apply AvgPool and MaxPool along the channel axis, concatenate them.
        - Feed forward through a Conv2D with filter size of $7\times7$
        - $M_s(F)=\sigma(f^{7\times7}([AvgPool(F);MaxPool(F)]))$
    - Combination
        
        - $F'=M_c(F)\otimes F$
        - $F''=M_s(F')\otimes F'$

```
import torch
from torch import nn
from torch.nn import functional as F

class CBAM:
    def __init__(self, 
                input_channels:int, 
                r:int):
        # self.gap_c = nn.AdaptiveAvgPool2d((1, 1))
        # self.gmp_c = nn.AdaptiveMaxPool2d((1, 1))

        self.fc1 = nn.Linear(input_channels, input_channels // r)
        self.fc2 = nn.Linear(input_channels // r, input_channels)
        self.mlp = nn.Sequential(
            # Can also achieve flatten by x.view(bc) (b, c, 1, 1) -> (b, c)
            # x will be gap_c and gmp_c
            nn.Flatten(),
            self.fc1,
            nn.ReLU(),
            self.fc2
        )
        # input channels = 2 since the tensor is a concatenated tensor with 2 single channel tensor
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=(7 - 1) // 2, bias=False)

    def channelAtt(self, x):
        b, c, h, w = x.size()
        # Approach 1:
        # max and average pooling along spatial axis
        gap_c = torch.sum(x, (2, 3)) / (h * w)
        # max function can apply to one axis only, values and indices are return at the same time
        gmp_c = torch.max(x.view(b, c, -1), 2)[0]

        # Approach 2:
        # gap_c = self.gap_c(x)
        # gmp_c = self.gmp_c(x)

        channel_att = F.sigmoid(self.mlp(gap_c) + self.mlp(gmp_c))
        return x * channel_att.view(b, c, 1, 1)

    def spatialAtt(self, x):
        # x -> (b, c, h, w)
        # max and average pooling along channel axis
        # and concatenate the two outputs along this axis (Channel)
        pool = torch.cat((torch.max(x, 1, keepdim=True)[0], torch.mean(x, 1, keepdim=True)), 1)
        spatial_att = self.conv(pool)
        spatial_att = F.sigmoid(spatial_att)
        return x * spatial_att

    def forward(self, x):
        x = self.channelAtt(x)
        return self.spatialAtt(x)
```