'''
Title
PVT v2: Improved baselines with Pyramid Vision Transformer

References
https://link.springer.com/10.1007/s41095-022-0274-8
'''



import torch
from torch import nn
from utils import PatchEmbeddingV1, Patch2Token, Token2Patch, Transformer, MergeHeads

class DWConv(nn.Module):
    def __init__(self, channels):
        super(DWConv, self).__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

    def forward(self, x, H, W):
        '''
        x should be in shape of (B, N, C)
        '''
        return Patch2Token(self.dw(Token2Patch(x, H, W)))

class Mlp(nn.Module):
    def __init__(self, in_channels, mid_channels=None, out_channels=None, 
                act_layer=nn.GELU, norm_layer=None, 
                bias=True, drop=0, use_conv=False):
        super(Mlp, self).__init__()
        out_channels = out_channels or in_channels
        mid_channels = mid_channels or in_channels
        self.fc1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias) if use_conv else nn.Linear(in_channels, mid_channels, bias)
        self.act1 = nn.ReLU()
        self.dw = DWConv(mid_channels)
        self.act2 = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(mid_channels) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Conv2d(mid_channels, out_channels, 1, bias=bias) if use_conv else nn.Linear(mid_channels, out_channels, bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dw(x, H, W)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Attention(nn.Module):
    def __init__(self, channels, num_heads=32, down_sample=True, qkv_bias=True, attn_drop=0, proj_drop=0):
        super(Attention, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        head_channels = channels // num_heads
        self.scale = head_channels ** -0.5
        self.q = nn.Linear(channels, channels, bias=qkv_bias)
        self.kv = nn.Linear(channels, channels * 2, bias=qkv_bias)

        self.drop = nn.Dropout(attn_drop)

        self.down_sample = down_sample
        if down_sample:
            self.down_sample = nn.Sequential(
                nn.AvgPool2d(7),
                nn.Conv2d(channels, channels, 1, )

            )
            self.down_norm = nn.LayerNorm(channels)
            self.down_act = nn.GELU()

        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.size()
        assert N == H * W

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale
        if self.down_sample:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = Patch2Token(self.down_sample(x))
            x = self.down_act(self.down_norm(x))
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        attn = q @ k.transpose(-2, -1)
        attn = self.drop(attn.softmax(dim=-1))

        # Merging heads
        x_attn = MergeHeads(attn @ v)
        x_attn = self.proj_drop(self.drop(x_attn))
        return x_attn

class PVTBlock(Transformer):
    def __init__(self, channels, num_heads=32, reduction_factor=1, mlp_ratio=4, qkv_bias=True,
                drop_path=0,
                attn_drop=0, proj_drop=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(PVTBlock, self).__init__(channels, mlp_ratio, act_layer, norm_layer, proj_drop, drop_path)
        self.token_mixer = Attention(channels, num_heads, reduction_factor, qkv_bias, attn_drop, proj_drop)
        self.mlp = Mlp(channels, act_layer=act_layer, norm_layer=norm_layer, drop=proj_drop)
    
class PVTv2Stage(nn.Module):
    def __init__(self, channels, embed_channels, stride, norm_pe=None,
                reduction=1, num_blocks=2, num_heads=4, mlp_ratio=4.0, qkv_bias=True, 
                attn_drop=0, proj_drop=0, drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(PVTv2Stage, self).__init__()
        if isinstance(reduction, int):
            reduction = [reduction] * num_blocks
        self.stride = stride
        # Overlapping patch embedding        
        self.pe = PatchEmbeddingV1(channels, embed_channels, 2 * stride - 1, stride, stride - 1, norm_pe)
        self.blocks = nn.ModuleList([
            PVTBlock(embed_channels, num_heads, reduction[idx], mlp_ratio, qkv_bias, drop_path,
                    attn_drop, proj_drop, act_layer, norm_layer) for idx in range(num_blocks)
        ])

    def forward(self, x):
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
    stage = PVTv2Stage(64, 128, 4, reduction=4)
    print(stage(t).size())
    stage = PVTv2Stage(64, 128, 5, reduction=4)
    print(stage(t).size())


