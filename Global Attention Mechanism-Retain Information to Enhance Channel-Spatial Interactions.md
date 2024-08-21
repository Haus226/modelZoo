Concept

- Overview:
    
    - The GAM module tries to compute attention based on cross-dimension interaction.
    - The module adopts the structure of CBAM ([Woo et al., 2018](zotero://select/library/items/TGYQCT58)). Why use sequential instead of parallel like BAM ([Park et al., 2018](zotero://select/library/items/BPLDE32D))?
    - The cross dimension interaction is achieved by a simple permutation of input features from $(N, C, H, W)\to(N,H,W,C)$
- Channel Attention:
    
    - Permute the input features and feed forward to a two layer encoder-decoder structure MLP with a reduction ratio $r$ without any spatial pooling (To retain the spatial information?)
    - The attention is then reverse permuted and reshaped, followed by a sigmoid function.
    - Finally, the output features is obtained by element-wise multiplication between the input features and the attention.
    - And this output features is passed to spatial attention submodule.
- Spatial Attention:
    
    - The input features are passed through two $7\times7$ convolution layers and activated using sigmoid function to obtain the attention.
    - Finally, the output features is obtained by element-wise multiplication between the input features and the attention.

```
import torch
from torch import nn
from torch.nn import functional as F

class GAM:
    def __init__(self, 
                mlp_input_channels:int, 
                conv_input_channels:int,
                r:int):
        self.fc1 = nn.Linear(mlp_input_channels, mlp_input_channels // r, bias=False)
        self.fc2 = nn.Linear(mlp_input_channels // r, mlp_input_channels, bias=False)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            self.fc1,
            nn.ReLU(),
            self.fc2
        )
        self.conv1 = nn.Conv2d(conv_input_channels, conv_input_channels // r, kernel_size=7, stride=1, padding=(7 - 1) // 2, bias=False)
        self.conv2 = nn.Conv2d(conv_input_channels // r, conv_input_channels, kernel_size=7, stride=1, padding=(7 - 1) // 2, bias=False)

    def channelAtt(self, x):
        b, c, h, w = x.size()
        x_permute = x.permute(0, 2, 3, 1)
        channel_att = F.sigmoid(self.mlp(x_permute).view(b, h, w, c).permute(0, 3, 1, 2))
        return x * channel_att

    def spatialAtt(self, x):
        spatial_att = self.conv1(x)
        spatial_att = F.sigmoid(self.conv2(spatial_att))
        return x * spatial_att

    def forward(self, x):
        x = self.channelAtt(x)
        return self.spatialAtt(x)
```