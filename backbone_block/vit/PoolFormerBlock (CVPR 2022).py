'''
Title
MetaFormer Is Actually What You Need for Vision

References
http://arxiv.org/abs/2111.11418
'''



import torch
from torch import nn
from utils import Mlp, PatchEmbeddingV1, GroupNorm
from timm.layers import DropPath

class Pooling(nn.Module):
    def __init__(self, pool_size=3):
        super(Pooling, self).__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x

class PoolFormerBlock(nn.Module):
    def __init__(self, channels, pool_size=3, mlp_ratio=4,
                act_layer=nn.GELU, norm_layer=GroupNorm,
                drop=0, drop_path=0, layer_scale=1e-5):
        super(PoolFormerBlock, self).__init__()

        self.norm1 = norm_layer(channels)
        self.token_mixer = Pooling(pool_size)
        self.norm2 = norm_layer(channels)
        self.mlp = Mlp(channels, int(mlp_ratio * channels), act_layer=act_layer, drop=drop, use_conv=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.layer_scale_1 = nn.Parameter(layer_scale * torch.ones(channels)) if layer_scale > 0 else 1
        self.layer_scale_2 = nn.Parameter(layer_scale * torch.ones(channels)) if layer_scale > 0 else 1

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.view(x.size(1), 1, 1) * self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.view(x.size(1), 1, 1) * self.mlp(self.norm2(x)))
        return x

class PoolFormerStage(nn.Module):
    def __init__(self, in_channels, patch_size, embed_channels, stride, padding=0, norm_pe=None, 
                num_blocks=2, pool_size=3, mlp_ratio=4,
                act_layer=nn.GELU, norm_layer=GroupNorm,
                drop=0, drop_path=0, layer_scale=1e-5):
        super(PoolFormerStage, self).__init__()
        self.pe = PatchEmbeddingV1(in_channels, embed_channels, patch_size, stride, padding, norm_pe)
        self.blocks = nn.ModuleList([
            PoolFormerBlock(embed_channels, pool_size, mlp_ratio, act_layer, norm_layer, drop, drop_path, layer_scale)
            for _ in range(num_blocks)
        ])
        

    def forward(self, x):
        '''
        Should be more blocks, this code is just for testing
        '''
        x = self.pe(x)
        for block in self.blocks:
            x = block(x)
        return x
    
if __name__ == "__main__":
    torch.manual_seed(226)
    t = torch.rand((32, 3, 224, 224))
    stage_1 = PoolFormerStage(3, 7, 64, 4, 3)
    t_s1 = stage_1(t)
    print(t_s1.size())
    stage_2 = PoolFormerStage(64, 3, 128, 2, 1)
    t_s2 = stage_2(t_s1)
    print(t_s2.size())
    stage_3 = PoolFormerStage(128, 3, 320, 2, 1)
    t_s3 = stage_3(t_s2)
    print(t_s3.size())
    stage_4 = PoolFormerStage(320, 3, 512, 2, 1)
    t_s4 = stage_4(t_s3)
    print(t_s4.size())