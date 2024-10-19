'''
Title
MaxViT: Multi-Axis Vision Transformer

References
http://arxiv.org/abs/2204.01697
'''



import torch
from torch import nn
from utils import Mlp, GP, RGP, WP, RWP, SEBlock, ConvBNReLU, rel_pos_idx
import torch.nn.functional as F
from timm.layers import DropPath

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion=4, r=4, drop=0):
        assert stride in [1, 2], "Stride must be either 1 or 2"
        super(MBConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.norm = nn.BatchNorm2d(in_channels)
        self.pw_conv = ConvBNReLU(in_channels, expansion * in_channels, 1, act=nn.GELU)
        self.dw_conv = ConvBNReLU(expansion * in_channels, expansion * in_channels, 3, stride, padding=1, groups=expansion * in_channels, act=nn.GELU)
        self.se = SEBlock(expansion * in_channels, r)
        self.proj = ConvBNReLU(expansion * in_channels, out_channels, 1, act=nn.GELU)
        if in_channels != out_channels:
            self.res_proj = ConvBNReLU(in_channels, out_channels, 1, act=nn.GELU)
        self.drop_path = DropPath(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        if self.in_channels != self.out_channels:
            residual = self.res_proj(residual)
            if self.stride == 2:
                residual = F.avg_pool2d(residual, kernel_size=3, stride=self.stride, padding=1)
        return residual + self.drop_path(self.proj(self.se(self.dw_conv(self.pw_conv(self.norm(x))))))
    
class Attention(nn.Module):
    def __init__(self, channels, window_size=7, num_heads=32,
                qkv_bias=True, attn_drop=0, proj_drop=0):
        super(Attention, self).__init__()
        self.window_size = window_size
        self.channels = channels
        self.num_heads = num_heads
        head_channels =channels // num_heads
        self.scale = head_channels ** -0.5

        self.rel_pos_bias = nn.Parameter(
            torch.rand(((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        )
        self.rel_pos_idx = rel_pos_idx(window_size[0], window_size[1])

        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B_, N, C = x.size()
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.chunk(3, dim=0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        rel_pos_bias = self.rel_pos_bias[self.rel_pos_idx].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1)
        attn = attn + rel_pos_bias.unsqueeze(0)
        attn = self.drop(attn.softmax(dim=-1))

        # (B_, H, N, N) @ (B_, H, N, C // H) -> (B_, H, N, C //H) -> (B_, N, H, C // H) -> (B_, N, C)
        x_attn = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x_attn = self.proj_drop(self.drop(x_attn))
        return x_attn


class MaxViTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion=4, r=4, drop=0,
                window_size=[7, 7], num_heads=32,
                qkv_bias=True, attn_drop=0, proj_drop=0):
        super(MaxViTBlock, self).__init__()
        self.window_size = window_size
        self.mb_conv = MBConv(in_channels, out_channels, stride, expansion, r, drop)
        
        self.block_pre_norm = nn.LayerNorm(out_channels)
        self.block_att = Attention(out_channels, window_size, num_heads, qkv_bias, attn_drop, proj_drop)
        self.block_post_norm = nn.LayerNorm(out_channels)
        self.block_mlp = Mlp(out_channels, drop)
        
        self.grid_pre_norm = nn.LayerNorm(out_channels)
        self.grid_att = Attention(out_channels, window_size, num_heads, qkv_bias, attn_drop, proj_drop)
        self.grid_post_norm = nn.LayerNorm(out_channels)
        self.grid_mlp = Mlp(out_channels, drop)
        

    def forward(self, x):
        # MBConv
        x = self.mb_conv(x)
        _, _, H, W = x.size()

        # Block Attention
        x = WP(x, self.window_size[0], self.window_size[1])
        x = self.block_att(self.block_pre_norm(x))
        x = self.block_post_norm(x)
        x = RWP(x, H, W, self.window_size[0], self.window_size[1])
        # (B, C, H, W) ---> (B, H, W, C) ---> (B, C, H, W)
        x = x + self.block_mlp(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # Grid Attention
        x = GP(x, self.window_size[0], self.window_size[1])
        x = self.grid_att(self.grid_pre_norm(x))
        x = self.grid_post_norm(x)
        x = RGP(x, H, W, self.window_size[0], self.window_size[1])
        # (B, C, H, W) ---> (B, H, W, C) ---> (B, C, H, W)
        x = x + self.grid_mlp(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x
    
class MaxViTStage(nn.Module):
    def __init__(self, in_channels, out_channels, stride, num_blocks=2, expansion=4, r=4, drop=0,
                window_size=[7, 7], num_heads=32,
                qkv_bias=True, attn_drop=0, proj_drop=0):
        self.blocks = nn.ModuleList([
            MaxViTBlock(in_channels, out_channels, 2, expansion, r, drop, 
                        window_size, num_heads, qkv_bias, attn_drop, proj_drop),
            *[MaxViTBlock(out_channels, out_channels, 1, expansion, r, drop, 
                        window_size, num_heads, qkv_bias, attn_drop, proj_drop) for _ in range(num_blocks - 1)]
        ])
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
    

if __name__ == "__main__":
    torch.manual_seed(226)
    t = torch.rand((32, 64, 21, 21))
    # Downsampling Test
    mb_conv = MBConv(64, 128, 2)
    print((mb_conv(t).size()))

    mv_block = MaxViTBlock(64, 128, 1)
    print((mv_block(t).size()))

    t = torch.rand((32, 64, 28, 28))
    mv_block = MaxViTBlock(64, 128, 2)
    print((mv_block(t).size()))
