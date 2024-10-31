'''
Title
An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

References
http://arxiv.org/abs/2010.11929
'''



import torch
from torch import nn
from einops.layers.torch import Rearrange
from utils import Mlp

class LinearPatching(nn.Module):
    def __init__(self, channels, embed_channels, patch_size):
        super(LinearPatching, self).__init__()
        patch_channels = channels * patch_size * patch_size
        self.patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_channels),
            nn.Linear(patch_channels, embed_channels),
            nn.LayerNorm(embed_channels)
        )
    
    def forward(self, x):
        return self.patch_embedding(x)

class Attention(nn.Module):
    def __init__(self, channels, num_heads=32, qkv_bias=True, attn_drop=0, proj_drop=0):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_channels = channels // num_heads
        self.scale = head_channels ** -0.5
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)

        self.drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.size()

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.chunk(3, dim=0)
        q = q * self.scale

        attn = q @ k.transpose(-2, -1)
        attn = self.drop(attn.softmax(dim=-1))

        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_attn = self.proj_drop(self.proj(x_attn))
        return x_attn

class ViTBlock(nn.Module):
    def __init__(self, channels, num_heads, mlp_ratio=4, qkv_bias=True, 
                proj_drop=0, attn_drop=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super(ViTBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.token_mixer = Attention(channels, num_heads, qkv_bias, attn_drop, proj_drop)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = Mlp(channels, int(channels * mlp_ratio), act_layer=act_layer, norm_layer=norm_layer, drop=proj_drop)

    def forward(self, x):
        x = x + self.token_mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
if __name__ == "__main__":
    torch.manual_seed(226)
    x = torch.rand((4, 24, 32, 32))
    
    pe = LinearPatching(24, 32, 4)
    p = pe(x)    
    B, N, C = p.size()

    # Concatenate class token and add position embedding
    pos = nn.Parameter(torch.rand(1, N + 1, C))
    cls_tokens = nn.Parameter(torch.rand(1, 1, C)).repeat(B, 1, 1)

    p = torch.cat([cls_tokens, p], dim=1) + pos

    vit = ViTBlock(32, 4)
    p = vit(p)
    
    print(p.size())
    # For classification take the first token i.e x[:, 0] or mean across the
    # token dimension i.e x.mean(dim=1)
    # Then the outputs are passed into a small mlp classification head.

