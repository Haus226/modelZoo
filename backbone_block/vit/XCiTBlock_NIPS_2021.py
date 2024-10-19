'''
Title
XCiT: Cross-Covariance Image Transformers

References
http://arxiv.org/abs/2106.09681
https://github.com/facebookresearch/xcit/blob/main/xcit.py#L20
'''



import torch
from torch import nn
import torch.nn.functional as F
from utils import Mlp, Token2Patch, Patch2Token
from timm.layers import DropPath

'''
Excluding the sinusodial position embedding
'''

class LPI(nn.Module):
    '''
    in_channels equals to out_channels in the paper while mid_channels is None
    '''
    def __init__(self, channels,
                kernel_size=3, stride=1, padding=0, dilation=1,
                act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, 
                ):
        super(LPI, self).__init__()
        self.dw1 = nn.Conv2d(channels, channels, kernel_size, stride, padding, dilation, channels)
        self.act = act_layer()
        self.norm = norm_layer(channels) if norm_layer is not None else nn.Identity()
        self.dw2 = nn.Conv2d(channels, channels, kernel_size, stride, padding, dilation, channels)

    def forward(self, x, H, W):
        assert x.size(1) == H * W 

        x = Token2Patch(x, H, W)
        x = self.dw1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.dw2(x)
        x = Patch2Token(x)
        return x

class XCAttention(nn.Module):
    def __init__(self, channels, num_heads=32, qkv_bias=True, attn_drop=0, proj_drop=0):
        super(XCAttention, self).__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.size()

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.chunk(3, dim=0)

        # (B, H, N, C) ---> (B, H, C, N)
        q = q.transpose(-1, -2)
        k = k.transpose(-1, -2)
        v = v.transpose(-1, -2)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # (B, H, C, N) @ (B, H, N, C) ---> (B, H, C, C)
        attn = (q @ k.transpose(-1, -2)) * self.temperature
        attn = attn.softmax(dim=-1)

        attn = self.drop(attn)

        # (B, H, C, C) @ (B, H, C, N) ---> (B, H, C, N) ---> (B, N, H, C)
        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)

        return x_attn

class XCABlock(nn.Module):
    def __init__(self, channels, num_heads=32, qkv_bias=True, attn_drop=0, proj_drop=0,
                kernel_size=3, stride=1, padding=0, dilation=1,
                mlp_ratio=4, act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale=0.0001, drop_path=0
                ):
        super(XCABlock, self).__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.xc_attn = XCAttention(channels, num_heads, qkv_bias, attn_drop, proj_drop)
        self.norm2 = nn.LayerNorm(channels)
        self.lpi = LPI(channels, kernel_size, stride, padding, dilation, act_layer)
        self.norm3 = nn.LayerNorm(channels)
        self.mlp = Mlp(channels, int(channels * mlp_ratio), act_layer=act_layer, norm_layer=norm_layer, drop=proj_drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma1 = nn.Parameter(layer_scale * torch.ones(channels)) if layer_scale > 0 else 1.0
        self.gamma2 = nn.Parameter(layer_scale * torch.ones(channels)) if layer_scale > 0 else 1.0
        self.gamma3 = nn.Parameter(layer_scale * torch.ones(channels)) if layer_scale > 0 else 1.0

    def forward(self, x, H, W):
        x = x + self.drop_path(self.gamma1 * self.xc_attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma2 * self.lpi(self.norm2(x), H, W))
        x = x + self.drop_path(self.gamma3 * self.mlp(self.norm3(x)))
        return x

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
        
if __name__ == "__main__":
    torch.manual_seed(226)
    x = torch.rand((4, 24, 32, 32))
    
    from utils import PatchEmbeddingV1
    pe = PatchEmbeddingV1(24, 32, 4, 4)
    p = pe(x)
    H_, W_ = p.shape[2:]
    p = Patch2Token(p)

    xca = XCABlock(32, 4, padding=1)
    p = xca(p, H_, W_)

    cls = torch.rand((4, 1, 32))
    p = torch.cat([cls, p], dim=1)

    class_att = ClassAttentionBlock(32, 4)
    p = class_att(p)

    print(p.size())


