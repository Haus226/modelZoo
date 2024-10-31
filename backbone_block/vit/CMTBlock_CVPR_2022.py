import torch
from torch import nn
from utils import Patch2Token, rel_pos_idx, PatchEmbeddingV1, Token2Patch, ConvBNAct, Transformer, GroupNorm
from math import floor

class LightMutilHeadSelfAttention(nn.Module):
    def __init__(self, channels, num_heads=32, spatial_size=8, reduction_factor=1, qkv_bias=True, attn_drop=0, proj_drop=0):
        super(LightMutilHeadSelfAttention, self).__init__()
        '''
        Here is the different point compared to other variants that use RPE. Most of them utilize
        window based attention which means (B, C, H, W) ---> (B * H // W_H * W // W_W, W_H * W_W, C), 
        the number of tokens becomes product of W_H and W_W.
        But, in this variant, we do not perform such or similar operation that rearrange the tensor,
        therefore the number of tokens will be H * W.
        '''

        self.spatial_size = (spatial_size, spatial_size) if isinstance(spatial_size, int) else spatial_size

        self.channels = channels
        self.num_heads = num_heads
        head_channels = channels // num_heads
        self.scale = head_channels ** -0.5
        self.q = nn.Linear(channels, channels, bias=qkv_bias)
        self.kv = nn.Linear(channels, channels * 2, bias=qkv_bias)

        self.rel_pos_bias = nn.Parameter(
            torch.rand(((2 * self.spatial_size[0] - 1) * (2 * self.spatial_size[1] - 1), num_heads))
        )
        self.rel_pos_idx = rel_pos_idx(self.spatial_size[0], self.spatial_size[1])
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
        k, v = kv[0], kv[1]
        '''
        The paper mentions that they are using relative position encoding but actually not in 
        the official repository, just absolute position encoding
        '''
        attn = q @ k.transpose(-2, -1)
        
        rel_pos_bias = self.rel_pos_bias[self.rel_pos_idx].view(
            self.spatial_size[0] * self.spatial_size[1], self.spatial_size[0] * self.spatial_size[1], -1)[:, :k.shape[2], :]
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()  
        attn = attn + rel_pos_bias.unsqueeze(0)
        attn = self.drop(attn.softmax(dim=-1))

        # Merging heads
        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_attn = self.proj_drop(self.drop(x_attn))
        return x_attn

class Mlp(nn.Module):
    def __init__(self, in_channels, mid_channels=None, out_channels=None, kernel_size=3,
                act_layer=nn.GELU):
        super(Mlp, self).__init__()
        out_channels = out_channels or in_channels
        mid_channels = mid_channels or in_channels
        self.pw_1 = ConvBNAct(in_channels, mid_channels, 1, act=act_layer)
        self.dw = ConvBNAct(mid_channels, mid_channels, kernel_size, padding=kernel_size // 2, act=act_layer)
        self.pw_2 = ConvBNAct(mid_channels, out_channels, 1, act=None)
    
    def forward(self, x):
        x = self.pw_1(x)
        x = x + self.dw(x)
        x = self.pw_2(x)
        return x
    
class CMTBlock(Transformer):
    def __init__(self, channels, spatial_size, reduction_factor=1, kernel_size=3, num_heads=4, mlp_ratio=4, act_layer=nn.GELU,
                qkv_bias=True, attn_drop=0, proj_drop=0, drop_path=0, layer_scale=0):
        super().__init__(channels, mlp_ratio, act_layer, nn.LayerNorm, proj_drop, drop_path, layer_scale)
        self.lpu = nn.Conv2d(channels, channels, kernel_size, 1, 1, groups=channels)
        self.token_mixer = LightMutilHeadSelfAttention(channels, num_heads, spatial_size, reduction_factor, 
                                                    qkv_bias, attn_drop, proj_drop)
        self.mlp = Mlp(channels, int(mlp_ratio * channels), kernel_size=kernel_size, act_layer=act_layer)
        self.norm_2 = GroupNorm(channels)

    def forward(self, x, H, W):
        x = self.lpu(x)
        x = Patch2Token(x)
        x = x + self.drop_path1(self.norm_1(self.token_mixer(x, H, W)))
        x = Token2Patch(x, H, W)
        x = x + self.drop_path2(self.norm_2(self.mlp(x)))
        return x
    
class CMTStage(nn.Module):
    def __init__(self, channels, embed_channels, spatial_size, patch_size, stride, padding=0, norm_pe=None,
                num_blocks=2, reduction_factor=2, num_heads=4, kernel_size=3, mlp_ratio=4.0, qkv_bias=True, 
                proj_drop=0, attn_drop=0, drop_path=0, act_layer=nn.GELU):
        super(CMTStage, self).__init__()
        self.patch_size = patch_size
        self.padding = padding
        self.stride = stride

        # For odd patch_size w or w/o padding and even patch_size with stride == patch_size
        spatial_size = [spatial_size, spatial_size] if isinstance(spatial_size, int) else spatial_size
        spatial_size = [
            floor((spatial_size[0] - patch_size + 2 * padding) / stride + 1),
            floor((spatial_size[1] - patch_size + 2 * padding) / stride + 1)
        ]
        self.pe = PatchEmbeddingV1(channels, embed_channels, patch_size, stride, padding, norm_pe)
        self.blocks = nn.ModuleList([
            CMTBlock(embed_channels, spatial_size, reduction_factor, kernel_size, num_heads, mlp_ratio,
            act_layer, qkv_bias, attn_drop, proj_drop, drop_path) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = self.pe(x)
        H_, W_ = x.shape[2:]
        for block in self.blocks:
            x = block(x, H_, W_)
        return x



if __name__ =="__main__":
    torch.manual_seed(226)
    t = torch.rand((8, 32, 32, 32))
    stage = CMTStage(32, 16, 32, 4, 4, 0, num_blocks=5)
    print(stage(t).size())
    stage = CMTStage(32, 16, 32, 5, 3, 7, num_blocks=5)
    print(stage(t).size())


