'''
Title
MixFormer: Mixing Features across Windows and Dimensions

References
https://ieeexplore.ieee.org/document/9878583/
https://github.com/PaddlePaddle/PaddleClas/blob/7b6c148065ba602dccf3e75a484f83e0097dcd30/ppcls/arch/backbone/model_zoo/mixformer.py
'''



import torch
from torch import nn
from utils import rel_pos_idx, WP, RWP, Mlp, PatchEmbeddingV2
from timm.layers import DropPath
import torch.nn.functional as F
import math

class MixingAttention(nn.Module):
    def __init__(self, channels, window_size, kernel_size, num_heads,
                qkv_bias=True, attn_drop=0, proj_drop=0):
        super(MixingAttention, self).__init__()
        self.channels = channels
        attn_channels = channels // 2
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        head_channels = attn_channels // num_heads
        self.scale = head_channels ** -0.5

        self.rel_pos_bias = nn.Parameter(
            torch.rand(((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        )
        self.rel_pos_idx = rel_pos_idx(window_size[0], window_size[1])

        # Projection layer for both branches
        self.proj_attn = nn.Linear(channels, channels // 2)
        self.proj_attn_norm = nn.LayerNorm(channels // 2)
        self.proj_cnn = nn.Linear(channels, channels)
        self.proj_cnn_norm = nn.LayerNorm(channels)

        # Conv branch
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=channels),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.BatchNorm2d(channels // 8),
            nn.GELU(),
            nn.Conv2d(channels // 8, channels // 2, kernel_size=1)
        )
        self.projection = nn.Conv2d(channels, channels // 2, kernel_size=1)
        self.conv_norm = nn.BatchNorm2d(channels // 2)

        # Window-Attention branch
        self.qkv = nn.Linear(channels // 2, channels // 2 * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(channels // 2, channels // 16, kernel_size=1),
            nn.BatchNorm2d(channels // 16),
            nn.GELU(),
            nn.Conv2d(channels // 16, 1, kernel_size=1)
        )
        self.attn_norm = nn.LayerNorm(channels // 2)

        # Final projection
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, h, w):
        '''
        x should have shape (B * H // win * W // win, win * win, C)
        '''

        # These projection layers are linear (can be replaced by conv with kernel size = 1) that project
        # the channels of input x

        # (B * H // win * W // win, win * win, C) ---> (B * H // win * W // win, win * win, C // 2)
        x_attn = self.proj_attn_norm(self.proj_attn(x))
        # (B * H // win * W // win, win * win, C) ---> (B * H // win * W // win, win * win, C)
        x_cnn = self.proj_cnn_norm(self.proj_cnn(x))

        # Reverse window partition to pass the input through Conv branch
        # (B * H // win * W // win, win * win, C) ---> (B, C, H, W)
        x_cnn = RWP(x, h, w, self.window_size[0], self.window_size[1])
        x_cnn = self.conv(x_cnn)
        # (B, C, H, W) ---> (B, C // 2, 1, 1)
        channel_interaction = self.channel_interaction(x_cnn)
        x_cnn = self.projection(x_cnn)

        # (B * H // win * W // win, win * win, C // 2), B_ = B * H // win * W // win
        B_, N, C = x_attn.size()
        # (B_, N, 3, H, C // H) ---> (3, B_, H, N, C // H)
        qkv = self.qkv(x_attn).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.chunk(3, dim=0)
        # (B, C // 2, 1, 1) ---> (B, 1, H, 1, (C // 2) // H)
        channel_interaction = channel_interaction.sigmoid().reshape(-1, 1, self.num_heads, 1, C // self.num_heads)
        # (B_, N, 3, H, C // H) ---> (B, , H, N, C // H)
        v = v.reshape(channel_interaction.size(0), -1, self.num_heads, N, C // self.num_heads)
        v = v * channel_interaction
        v = v.reshape(q.size())

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        
        rel_pos_bias = self.rel_pos_bias[self.rel_pos_idx].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1)
        attn = attn + rel_pos_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (B_, H, N, N) @ (B_, H, N, C // H) ---> (B_, H, N, C // H) ---> (B_, N, H, C // H) ---> (B_, N, C)
        # Merging heads
        x_attn = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        # Reverse window partition to integrate spatial interaction
        x_spatial = RWP(x_attn, h, w, self.window_size[0], self.window_size[1])
        spatial_interaction = self.spatial_interaction(x_spatial)
        x_cnn = (spatial_interaction * x_cnn).sigmoid()
        x_cnn = self.conv_norm(x_cnn)
        x_cnn = WP(x_cnn, self.window_size[0], self.window_size[1])

        # Concatenation
        x_attn = self.attn_norm(x_attn)
        x = torch.cat([x_attn, x_cnn], dim=-1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MixFormerBlock(nn.Module):
    def __init__(self, channels, num_heads, window_size, kernel_size, mlp_ratio, qkv_bias=True, 
                proj_drop=0, attn_drop=0, drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                ):
        super(MixFormerBlock, self).__init__()
        self.window_size = window_size
        self.norm1 = norm_layer(channels)
        self.token_mixer = MixingAttention(channels, window_size, kernel_size, num_heads, qkv_bias, attn_drop, proj_drop)
        self.norm2 = norm_layer(channels)
        self.mlp = Mlp(channels, int(mlp_ratio * channels), act_layer=act_layer, drop=proj_drop, use_conv=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, H, W):
        B, N, C = x.size()

        assert N == H * W
        residual = x
        x = self.norm1(x)
        x = x.reshape(B, C, H, W)

        pad_width = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_height = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]

        # Slightly different padding from official implementation
        padding = [pad_width // 2, math.ceil(pad_width / 2), pad_height // 2, math.ceil(pad_height / 2)]
        x = F.pad(x, padding)
        x_windows = WP(x, self.window_size[0], self.window_size[1])
        attn_windows = self.token_mixer(x_windows, *x.shape[2:] )
        x_reversed = RWP(attn_windows, x.size(2), x.size(3),
                        self.window_size[0], self.window_size[1])
        # Excluding the padded area
        x = x_reversed[:, :, padding[2]:-padding[3], padding[0]:-padding[1]]
        # (B, C, H, W) ---> (B, H * W, C)
        x = x.reshape([B, H * W, C])
        x = residual + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class MixFormerStage(nn.Module):
    def __init__(self, channels, embed_channels, patch_size, stride, padding=0, norm_pe=None,
                num_heads=4, window_size=[7, 7], kernel_size=3, mlp_ratio=4.0, qkv_bias=True, 
                proj_drop=0, attn_drop=0, drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(MixFormerStage, self).__init__()
        self.patch_size = patch_size
        self.padding = padding
        self.stride = stride
        self.pe = PatchEmbeddingV2(channels, embed_channels, patch_size, stride, padding, norm_pe)
        self.block = MixFormerBlock(embed_channels, num_heads, window_size, kernel_size, mlp_ratio,
                                    qkv_bias, proj_drop, attn_drop, drop_path, act_layer, norm_layer)

    def forward(self, x, H, W):
        # For even patch_size and stride == patch_size, then, H' = H // stride and W' = W //stride
        return self.block(self.pe(x, H, W), 
                          math.floor((H - self.patch_size + 2 * self.padding) // self.stride + 1), 
                          math.floor((W - self.patch_size + 2 * self.padding) // self.stride + 1), 
                )

if __name__ == "__main__":
    torch.manual_seed(226)
    t = torch.rand((32, 32 * 32, 64))
    block = MixFormerBlock(64, 4, [3, 3], 3, 4)
    print(block(t, 32, 32).size())
    stage = MixFormerStage(64, 128, 4, 4, 0)
    print(stage(t, 32, 32).size())
    stage = MixFormerStage(64, 128, 5, 2, 2)
    print(stage(t, 32, 32).size())