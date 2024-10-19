'''
Title
WeightNet: Revisiting the Design Space of Weight Networks

References
https://arxiv.org/abs/2007.11823
https://github.com/megvii-model/WeightNet
'''



import torch
from torch import nn
import torch.nn.functional as F

class WeightConv(nn.Module):
    def __init__(self, in_channels, out_channels, r, kernel_size, M, G, stride): 
        super(WeightConv, self).__init__()
        
        # M = 1, G = 1 --> SENet
        # M = num_experts / C_in, G = 1 / C_in --> CondConv
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // r, 1, stride=stride)
        # When M = m / C_in, this becomes the fc in CondConv, (B, m, 1, 1), attention of M filters
        # When M = 1, this becomes the fc in SENet, (B, C_in, 1, 1), attention of C input channels
        # Else this becomes the attention of C input channels of M filters I guess
        self.fc2 = nn.Conv2d(in_channels // r, M * in_channels, 1, bias=True)

        # Grouped Fully Connnected Layer
        # When M = m / C_in, G = 1 / C_in, this becomes the 
        
        # When M = 1, G = 1, after grouping, kernel has the shape of (C_out * K * K, 1, 1, 1) convole with (B, 1, 1, 1)
        # Ex: C_in = 2, C_out = 3, K = 3
        # [alpha1]--->[1|2|3|...|25|26|27]--->[alpha1|2alpha1|...|26alpha1|27alpha1] 
        # [alpha2]--->[1|2|3|...|25|26|27]--->[alpha2|2alpha2|...|26alpha2|27alpha2] 
        # Concatenate the output
        # [alpha1|2alpha1|...|26alpha1|27alpha1|alpha2|2alpha2|...|26alpha2|27alpha2]
        # Reshape from (54) ---> (3, 2, 3, 3) (C_out, C_in, K, K)
        # First view as (C_in, C_out, K, K) then permute (1, 0, 2, 3)? else the weight of second channel is incorrect?
        
        self.fc3 = nn.Conv2d(M * in_channels, in_channels * out_channels * kernel_size * kernel_size, 1, 
                            bias=False, groups=G * in_channels)
        self.sigmoid = nn.Sigmoid()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.M, self.G = M, G
        self.stride = stride
        self.padding = kernel_size // 2
        self.kernel_size = kernel_size

    def forward(self, x):
        B, C, H, W = x.size()
        weight = self.fc1(self.gap(x))  # (B, C, H, W) -> (B, C, 1, 1) -> (B, C // R, 1, 1)
        weight = self.fc2(weight)   # (B, C // R, 1, 1) -> (B, M * C, 1, 1)
        weight = self.sigmoid(weight) # The activation vector which has (M * C) dimension for each B
        weight = self.fc3(weight) # (B, M * C, 1, 1) -> (B, C * OC * K * K, 1, 1)
        x = x.view(1, B * C, H, W)
        # weight_ = weight.view(B * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        weight = weight.view(self.in_channels, B * self.out_channels, self.kernel_size, self.kernel_size).permute(1, 0, 2, 3)
        x = F.conv2d(x, weight=weight, stride=self.stride, padding=self.padding, groups=B)
        x = x.view(B, self.out_channels, H, W)
        return x