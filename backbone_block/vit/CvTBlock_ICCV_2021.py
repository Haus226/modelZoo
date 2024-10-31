'''
Title
CvT: Introducing Convolutions to Vision Transformers

References
http://arxiv.org/abs/2103.15808
'''



import torch
from torch import nn
from utils import Mlp, PatchEmbeddingV1, Token2Patch, Patch2Token, MergeHeads
from einops.layers.torch import Rearrange
from einops import rearrange
from timm.layers import DropPath


class DWConvFlatten(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DWConvFlatten, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            Rearrange("b c h w -> b (h w) c")
        )

    def forward(self, x):
        return self.block(x)

class Attention(nn.Module):
    def __init__(self, channels, num_heads, kernel_size, kv_stride, qkv_bias, attn_drop=0, drop=0):
        super(Attention, self).__init__()
        head_channels = channels // num_heads
        self.scale = head_channels ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)

        self.q = DWConvFlatten(channels, channels, kernel_size, 1, kernel_size // 2)
        self.kv = DWConvFlatten(channels, channels * 2, kernel_size, kv_stride, kernel_size // 2)
        self.proj_q = nn.Linear(channels, channels, bias=qkv_bias)
        self.proj_k = nn.Linear(channels, channels, bias=qkv_bias)
        self.proj_v = nn.Linear(channels, channels, bias=qkv_bias)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Dropout(drop)
        )
        self.num_heads = num_heads

    def forward(self, x, H, W):
        _, N, _ = x.size()
        assert N == H * W
        x = Token2Patch(x, H, W)

        # The conv including rearrange (b, c, h, w) to (b, h, w, c) already
        conv_q = self.q(x)
        conv_k, conv_v = torch.chunk(self.kv(x), 2, dim=-1)

        q = rearrange(self.proj_q(conv_q), 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(self.proj_k(conv_k), 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(self.proj_v(conv_v), 'b n (h d) -> b h n d', h=self.num_heads)
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = MergeHeads(attn @ v)
        return x

class CvTBlock(nn.Module):
    def __init__(self, channels, num_heads, kernel_size, kv_stride, qkv_bias, mlp_ratio=4,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                attn_drop=0, drop=0, drop_path=0):
        super(CvTBlock, self).__init__()
        self.norm1 = norm_layer(channels)
        self.token_mixer = Attention(channels, num_heads, kernel_size, kv_stride, qkv_bias, attn_drop, drop)
        self.norm2 = norm_layer(channels)
        self.mlp = Mlp(channels, int(channels * mlp_ratio), act_layer=act_layer, norm_layer=norm_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.token_mixer(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CvTStage(nn.Module):
    def __init__(self, channels, embed_channels, patch_size, stride, padding=0, norm_pe=None,
                num_blocks=2, num_heads=4, kernel_size=3, mlp_ratio=4.0, kv_stride=2, qkv_bias=True, 
                proj_drop=0, attn_drop=0, drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(CvTStage, self).__init__()
        self.patch_size = patch_size
        self.padding = padding
        self.stride = stride
        self.pe = PatchEmbeddingV1(channels, embed_channels, patch_size, stride, padding, norm_pe)
        self.blocks = nn.ModuleList([
            CvTBlock(embed_channels, num_heads, kernel_size, kv_stride, mlp_ratio,
            qkv_bias, act_layer, norm_layer, attn_drop, proj_drop, drop_path) for _ in range(num_blocks)
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
    block = Attention(64, 8, 3, 2, True)
    print(block(t, 32, 32).size())

    t = torch.rand((32, 64, 32, 32))
    stage = CvTStage(64, 128, 4, 4, 0)
    print(stage(t).size())
    stage = CvTStage(64, 128, 5, 2, 2)
    print(stage(t).size())

