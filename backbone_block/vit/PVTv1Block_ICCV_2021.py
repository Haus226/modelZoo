'''
Title
Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions

References
https://arxiv.org/abs/2102.12122
'''



import torch
from torch import nn
from utils import Mlp, PatchEmbeddingV1, Patch2Token
from einops import rearrange
from timm.layers import DropPath

class Attention(nn.Module):
    def __init__(self, channels, num_heads=32, reduction_factor=1, qkv_bias=True, attn_drop=0, proj_drop=0):
        super(Attention, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        head_channels = channels // num_heads
        self.scale = head_channels ** -0.5
        self.q = nn.Linear(channels, channels, bias=qkv_bias)
        self.kv = nn.Linear(channels, channels * 2, bias=qkv_bias)

        self.drop = nn.Dropout(attn_drop)

        self.reduce_factor = reduction_factor
        if self.reduce_factor > 1:
            self.down_sample = nn.Conv2d(channels, channels, reduction_factor, reduction_factor)
            self.down_norm = nn.LayerNorm(channels)

        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.size()
        assert N == H * W

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale
        if self.reduce_factor > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = Patch2Token(self.down_sample(x))
            x = self.down_norm(x)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.chunk(2, dim=0)
        
        attn = q @ k.transpose(-2, -1)
        attn = self.drop(attn.softmax(dim=-1))

        # Merging heads
        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_attn = self.proj_drop(self.drop(x_attn))
        return x_attn

class PVTBlock(nn.Module):
    def __init__(self, channels, num_heads=32, reduction_factor=1, mlp_ratio=4, qkv_bias=True,
                drop=0, drop_path=0,
                attn_drop=0, proj_drop=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(PVTBlock, self).__init__()
        self.norm1 = norm_layer(channels)
        self.token_mixer = Attention(channels, num_heads, reduction_factor, qkv_bias, attn_drop, proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(channels)
        self.mlp = Mlp(channels, mid_channels=int(mlp_ratio * channels), act_layer=act_layer, norm_layer=norm_layer, drop=drop)


    def forward(self, x, H, W):
        B, N, C = x.size()
        assert N == H * W
        x = x + self.drop_path(self.token_mixer(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
    
class PVTv1Stage(nn.Module):
    def __init__(self, channels, embed_channels, patch_size, stride, padding=0, norm_pe=None,
                reduction=1, num_blocks=2, num_heads=4, mlp_ratio=4.0, qkv_bias=True, 
                attn_drop=0, proj_drop=0, drop=0, drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(PVTv1Stage, self).__init__()
        if isinstance(reduction, int):
            reduction = [reduction] * num_blocks
        self.patch_size = patch_size
        self.padding = padding
        self.stride = stride
        self.pe = PatchEmbeddingV1(channels, embed_channels, patch_size, stride, padding, norm_pe)
        self.blocks = nn.ModuleList([
            PVTBlock(embed_channels, num_heads, reduction[idx], mlp_ratio, qkv_bias, drop, drop_path,
                    attn_drop, proj_drop, act_layer, norm_layer) for idx in range(num_blocks)
        ])

    def forward(self, x):
        # For even patch_size and stride == patch_size, then, H' = H // stride and W' = W //stride
        B, C, H, W = x.size()
        x = self.pe(x)
        H_, W_ = x.shape[2:]
        x = self.blocks[0](Patch2Token(x), H_, W_)
        for block in self.blocks[1:]:
            x = block(x, H_, W_)
        return x

if __name__ == "__main__":
    torch.manual_seed(226)
    t = torch.rand((32, 32 * 32, 64))
    pvtblock = PVTBlock(64, 2)
    print(pvtblock(t, 32, 32).size())

    t = torch.rand((32, 64, 32, 32))
    stage = PVTv1Stage(64, 128, 4, 4, 0, reduction=4)
    print(stage(t).size())
    stage = PVTv1Stage(64, 128, 5, 2, 2, reduction=4)
    print(stage(t).size())


