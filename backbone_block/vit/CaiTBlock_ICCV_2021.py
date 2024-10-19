'''
Title
Going deeper with Image Transformers

References
http://arxiv.org/abs/2103.17239
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/cait.py
'''



import torch
from torch import nn
from timm.layers import DropPath
from utils import Mlp

class ClassAttention(nn.Module):
    def __init__(self, channels, num_heads, qkv_bias=False, attn_drop=0, proj_drop=0):
        super(ClassAttention, self).__init__()
        self.num_heads = num_heads
        head_channels = channels // num_heads
        self.scale = head_channels ** -0.5

        self.q = nn.Linear(channels, channels, bias=qkv_bias)
        self.kv = nn.Linear(channels, channels * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        '''
        x is assumed to be the tokens concatenate with the class token
        '''
        B, N, C = x.size()
        q = self.q(x[:, 0]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.chunk(2, dim=0)

        attn_cls = (q @ k.transpose(-2, -1)) * self.scale
        attn_cls = self.attn_drop(attn_cls.softmax(dim=-1))

        attn_cls = (attn_cls @ v).transpose(1, 2).reshape(B, 1, C)
        attn_cls = self.proj(attn_cls)
        # Slightly different between Class Attention in XCiT and CaiT in timm module
        x = torch.cat([self.proj_drop(attn_cls), x[:, 1:]], dim=1)
        return x
    
class ClassAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads, mlp_ratio=4., qkv_bias=False, drop=0,
                attn_drop=0, drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                layer_scale=0.0001,
                ):
        super(ClassAttentionBlock, self).__init__()
        self.norm1 = nn.LayerNorm(channels)

        self.attn = ClassAttention(channels, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(channels)
        self.mlp = Mlp(channels, int(channels * mlp_ratio), act_layer=act_layer, norm_layer=norm_layer, drop=drop)

        self.gamma1 = nn.Parameter(layer_scale * torch.ones(channels)) if layer_scale > 0 else 1.0
        self.gamma2 = nn.Parameter(layer_scale * torch.ones(channels)) if layer_scale > 0 else 1.0

    def forward(self, x):
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        x_res = x
        # Use 0:1 to return the tensor in shape (B, 1, C) otherwise the returned tensor will
        # have shape (B, C)
        cls_token = self.gamma2 * self.mlp(x[:, 0:1])
        x = torch.cat([cls_token, x[:, 1:]], dim=1)
        x = x_res + self.drop_path(x)
        return x
    
class TalkingHeadAttention(nn.Module):
    def __init__(self, channels, num_heads=32, qkv_bias=True, attn_drop=0, proj_drop=0):
        super(TalkingHeadAttention, self).__init__()
        self.num_heads = num_heads
        head_channels = channels // num_heads
        self.scale = head_channels ** -0.5
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)

        self.drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

        # Approach from timm
        self.proj_pre = nn.Linear(num_heads, num_heads)
        self.proj_post = nn.Linear(num_heads, num_heads)

        # For approach from lucidrains, the above is the same if 
        # einsum(x, weight, "b h n c, g h -> b g n c")

    def forward(self, x):
        B, N, C = x.size()

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.chunk(3, dim=0)
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1)).squeeze()
        # (B, H, N, C) ---> (B, N, C, H) ---> (B, H, N, C)
        attn = self.proj_pre(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.drop(attn.softmax(dim=-1))
        attn = self.proj_post(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_attn = self.proj_drop(self.proj(x))
        return x_attn
    
class THAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads, mlp_ratio=4., qkv_bias=False, drop=0,
                attn_drop=0, drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                layer_scale=0.0001,
                ):
        super(THAttentionBlock, self).__init__()
        self.norm1 = nn.LayerNorm(channels)

        self.attn = TalkingHeadAttention(channels, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(channels)
        self.mlp = Mlp(channels, int(channels * mlp_ratio), act_layer=act_layer, norm_layer=norm_layer, drop=drop)

        self.gamma1 = nn.Parameter(layer_scale * torch.ones(channels)) if layer_scale > 0 else 1.0
        self.gamma2 = nn.Parameter(layer_scale * torch.ones(channels)) if layer_scale > 0 else 1.0

    def forward(self, x):
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x

if __name__ == "__main__":
    torch.manual_seed(226)
    x = torch.rand((4, 24, 32, 32))
    
    from utils import PatchEmbeddingV1, Patch2Token
    pe = PatchEmbeddingV1(24, 32, 4, 4)
    p = pe(x)
    p = Patch2Token(p)

    cait = THAttentionBlock(32, 4)
    p = cait(p)

    cls = torch.rand((4, 1, 32))
    p = torch.cat([cls, p], dim=1)

    class_att = ClassAttentionBlock(32, 4)
    p = class_att(p)

    print(p.size())
