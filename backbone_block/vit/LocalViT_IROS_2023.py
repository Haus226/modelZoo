import torch
from torch import nn
from utils import SEBlock, ConvBNAct, Transformer, Attention, Token2Patch, Patch2Token, PatchEmbeddingV1
from math import log
from typing import Union, Optional

class H_Sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(H_Sigmoid, self).__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu6(x + 3) / 6

class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, bias=1, act=nn.Sigmoid):
        super(ECABlock, self).__init__()
        kernel_size = int(abs((log(channels, 2) + bias) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        padding = kernel_size // 2
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=padding, bias=False)
        self.act = act()
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.avg(x)
        # (B, C, 1, 1) ---> (B, C, 1) ---> (B, 1, C)
        x = self.conv(x.squeeze(-1).transpose(-1, -2))
        x = x.transpose(-1, -2).unsqueeze(-1)
        x = self.act(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_channels, mid_channels=None, out_channels=None, kernel_size=3,
                act_layer=nn.Hardswish, 
                attn:Optional[Union[SEBlock, ECABlock]]=SEBlock, attn_act=H_Sigmoid, 
                **attn_kwargs):
        '''
        attn_kwargs: 
            For SEBlock should be an integer parameter r, the reduction factor of channels.
            For ECABlock should be two integers parameters, the gamma and bias
        '''
        super(Mlp, self).__init__()
        out_channels = out_channels or in_channels
        mid_channels = mid_channels or in_channels
        self.pw_1 = ConvBNAct(in_channels, mid_channels, 1, act=act_layer)
        self.dw = ConvBNAct(mid_channels, mid_channels, kernel_size, padding=kernel_size // 2, act=act_layer)
        self.pw_2 = ConvBNAct(mid_channels, out_channels, 1, act=None)
        self.attn = attn(channels=mid_channels, act=attn_act, **attn_kwargs) if attn else nn.Identity()
    
    def forward(self, x):
        res = x
        x = self.pw_1(x)
        x = self.dw(x)
        x = self.attn(x)
        x = self.pw_2(x)
        x = res + x
        return x

class LocalViTBlock(Transformer):
    def __init__(self, channels, num_heads, qkv_bias=True, attn_drop=0, mlp_ratio=4,
                kernel_size=3,  
                act_layer=nn.Hardswish, mlp_attn=SEBlock, 
                attn_act=H_Sigmoid,
                proj_drop=0, drop_path=0, 
                **attn_kwargs):
        super().__init__(channels, mlp_ratio, act_layer, nn.LayerNorm, proj_drop, drop_path, 0)
        self.token_mixer = Attention(channels, num_heads, qkv_bias, attn_drop, proj_drop)
        self.mlp = Mlp(channels, int(channels * mlp_ratio), kernel_size=kernel_size, 
                    act_layer=act_layer, attn=mlp_attn, attn_act=attn_act, **attn_kwargs)
        
    def forward(self, x, H, W):
        x = x + self.drop_path1(self.norm_1(self.token_mixer(x)))
        cls_tokens, x = x[:, :1], x[:, 1:]
        x = Token2Patch(x, H, W)
        x = self.mlp(x)
        x = Patch2Token(x)
        return torch.cat([cls_tokens, x], dim=1)
    
if __name__ == "__main__":
    torch.manual_seed(226)
    x = torch.rand((4, 24, 32, 32))
    
    pe = PatchEmbeddingV1(24, 32, 4, 4)
    p = pe(x)    
    H_, W_ = p.shape[2:]
    p = Patch2Token(p)
    B, N, C = p.size()

    # Concatenate class token and add position embedding
    pos = nn.Parameter(torch.rand(1, N + 1, C))
    cls_tokens = nn.Parameter(torch.rand(1, 1, C)).repeat(B, 1, 1)

    p = torch.cat([cls_tokens, p], dim=1) + pos

    localvit = LocalViTBlock(32, 4, r=2)
    p = localvit(p, H_, W_)
    print(p.size())

    localvit = LocalViTBlock(32, 4, bias=3, gamma=4, mlp_attn=ECABlock)
    p = localvit(p, H_, W_)
    print(p.size())
