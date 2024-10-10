'''
Title
Focal Modulation Networks

References
http://arxiv.org/abs/2203.11926
https://github.com/microsoft/FocalNet/blob/main/classification/focalnet.py
'''



import torch
from torch import nn
from timm.layers import DropPath

class FocalModulation(nn.Module):
    def __init__(self, channels, focal_window, focal_level, focal_factor, p, bias=True):
        super(FocalModulation, self).__init__()
        self.proj_in = nn.Linear(channels, 2 * channels + focal_level + 1, bias=bias)
        self.h = nn.Conv2d(channels, channels, 1, bias=bias)
        self.act = nn.GELU()
        self.proj_out = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Dropout(p)
        )

        self.focal_block = nn.ModuleList()
        for l in range(focal_level):
            kernel_size = focal_factor * l + focal_window
            self.focal_block.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size, groups=channels, 
                              padding=kernel_size // 2, bias=False),
                    nn.GELU()
                )
            )
    
    def forward(self, x):
        '''
        x should have the shape (B, H, W, C)
        '''
        C = x.size(-1)
        x = self.proj_in(x).permute(0, 3, 1, 2) # BHWC ---> BCHW
        q, z_pre, gates = torch.split(x, (C, C, len(self.focal_block) + 1), 1)

        z_cur = 0
        for l in range(len(self.focal_block)):
            z_pre = self.focal_block[l](z_pre)
            z_cur = z_cur + z_pre * gates[:, l:l + 1]
        z_cur = z_cur + self.act(z_pre.mean((2, 3), keepdim=True)) * gates[:, len(self.focal_block):]

        modulator = self.h(z_cur)
        y = q * modulator
        y = y.permute(0, 2, 3, 1)   # BHWC
        return self.proj_out(y)

class FocalNetBlock(nn.Module):
    def __init__(self, channels, focal_window, focal_level, focal_factor, focal_p=0, 
                mlp_layer=nn.Identity, 
                p=0, layer_scale=1e-4):
        '''
        Can be implemented as official repo which output (B, H * W, C) and using nn.LayerNorm
        '''
        super(FocalNetBlock, self).__init__()
        self.channels = channels
        self.pre_norm = nn.GroupNorm(1, channels)
        self.modulation = FocalModulation(channels, focal_window, focal_level, focal_factor, focal_p)
        self.post_norm = nn.GroupNorm(1, channels)
        self.mlp = mlp_layer()
        self.drop_path = DropPath(p)

        # Layer scaling proposed in "Going deeper with Image Transformers" (CaiT)
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(channels), 
                                    requires_grad=True) if layer_scale > 0 else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(channels), 
                                    requires_grad=True) if layer_scale > 0 else 1

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1.view(x.size(1), 1, 1) * self.modulation(self.pre_norm(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        x = x + self.drop_path(self.gamma_2.view(x.size(1), 1, 1) * self.mlp(x))
        return x

if __name__ == "__main__":
    torch.manual_seed(226)
    t = torch.rand((32, 21, 21, 64))
    f = FocalModulation(64, 1, 3, 2, 0.5, True)
    print(f(t).size())
    t = t.permute(0, 3, 1, 2)
    focalnet = FocalNetBlock(64, 1, 3, 2, 0.5, nn.Identity, 0.5)
    print(focalnet(t).size())

