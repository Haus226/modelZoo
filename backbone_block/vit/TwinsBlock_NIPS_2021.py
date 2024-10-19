'''
Title
Twins: Revisiting the Design of Spatial Attention in Vision Transformers

References
http://arxiv.org/abs/2104.13840
'''



import torch
from torch import nn
from utils import Mlp, CPE, PatchEmbeddingV1, Patch2Token
from einops import rearrange
from timm.layers import DropPath
import math
import torch.nn.functional as F

class LSA(nn.Module):
    def __init__(self, channels, window_size=7, num_heads=32,
                qkv_bias=True, attn_drop=0, proj_drop=0):
        super(LSA, self).__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        head_channels =channels // num_heads
        self.scale = head_channels ** -0.5

        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.size()
        assert N == H * W

        x = x.view(B, H, W, C)
        # assert (H % self.window_size[0]) == 0
        # assert (W % self.window_size[1]) == 0

        # Slightly different from official implementation, we perform padding
        pad_width = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_height = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]

        padding = [pad_width // 2, math.ceil(pad_width / 2), pad_height // 2, math.ceil(pad_height / 2)]
        x = F.pad(x, padding)
        h, w = x.shape[1:3]
        h_patch, w_patch = h // self.window_size[0], w // self.window_size[1]
        total_patches = h_patch * w_patch
        x = x.reshape(B, h_patch, self.window_size[0], w_patch, self.window_size[1], C).transpose(2, 3)

        qkv = self.qkv(x).reshape(B, total_patches, -1, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = torch.chunk(qkv, 3, dim=0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = self.drop(attn.softmax(dim=-1))

        # (B, P, H, N, N) @ (B, P, H, N, C // H) -> (B, P, H, N, C // H) -> (B, P, N, H, C // H) -> 
        # (B, H_P, W_P, WIN_H, WIN_W, C) -> (B, H_P, WIN_H, W_P, WIN_W, C)
        x_attn = rearrange((attn @ v).squeeze(), "b (p q) h (wh ww) c -> b (p wh) (q ww) (h c)", 
                        p=h_patch, q=w_patch, wh=self.window_size[0], ww=self.window_size[1])
        _, _, padded_h, padded_w = x_attn.size()
        # Excluded padded area
        x_attn = x_attn[:, :, padding[2]:padded_h - padding[3], padding[0]:padded_w - padding[1]]
        x_attn = x_attn.flatten(start_dim=1, end_dim=2)
        x_attn = self.proj_drop(self.drop(x_attn))
        return x_attn

class GSA(nn.Module):
    def __init__(self, channels, num_heads=32, window_size=1, qkv_bias=True, attn_drop=0, proj_drop=0):
        super(GSA, self).__init__()
        self.num_heads = num_heads
        head_channels = channels // num_heads
        self.scale = head_channels ** -0.5
        self.q = nn.Linear(channels, channels, bias=qkv_bias)
        self.kv = nn.Linear(channels, channels * 2, bias=qkv_bias)

        self.drop = nn.Dropout(attn_drop)
        # This is basically taken from PVT which summarize the important information 
        # for each of m × n sub-windows and the representative is used to communicate with other sub-windows 
        # (serving as the key in self-attention)
        self.window_size = window_size
        if self.window_size > 1:
            self.down_sample = nn.Conv2d(channels, channels, window_size, window_size)
            self.down_norm = nn.LayerNorm(channels)

        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.size()
        assert N == H * W
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale
        if self.window_size > 1:
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

class TwinsBlock(nn.Module):
    def __init__(self, channels, num_heads, global_window, local_window, mlp_ratio, qkv_bias=True, 
                proj_drop=0, attn_drop=0, drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                type="lsa"
            ):
        super(TwinsBlock, self).__init__()
        self.local_window = local_window
        self.global_window = global_window

        self.pre_norm = nn.LayerNorm(channels)
        self.token_mixer = LSA(channels, local_window, num_heads, qkv_bias, attn_drop, proj_drop) if type =="lsa" else  GSA(channels, global_window, num_heads, qkv_bias, attn_drop, proj_drop)
        self.post_norm = nn.LayerNorm(channels)
        self.mlp = Mlp(channels, int(mlp_ratio * channels), act_layer=act_layer, norm_layer=norm_layer, drop=proj_drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W):
        x = x + self.drop_path1(self.token_mixer(self.pre_norm(x), H, W))
        x = x + self.drop_path2(self.mlp(self.post_norm(x)))
        return x

class TwinsStage(nn.Module):
    '''
    Set all blocks to GSA for Twins-PCPVT
    '''
    def __init__(self, in_channels, embed_channels, patch_size, stride, padding, types,
                cpe_kernel=3, cpe_stride=1, cpe_padding=1, cpe_groups=1,
                num_heads=8, global_window=4, local_window=[7, 7], mlp_ratio=4, qkv_bias=True, 
                proj_drop=0, attn_drop=0, drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm
                ):
        super(TwinsStage, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.pe = PatchEmbeddingV1(in_channels, embed_channels, patch_size, stride, padding)
        self.cpe = CPE(embed_channels, embed_channels, cpe_kernel, cpe_stride, cpe_padding, cpe_groups)
        self.blocks = nn.ModuleList([
            TwinsBlock(embed_channels, num_heads, global_window, local_window, mlp_ratio, 
                    qkv_bias, proj_drop, attn_drop, drop_path, act_layer, norm_layer, types[idx]) for idx in range(len(types))]
        )

    def forward(self, x):
        B, C, H, W = x.size()
        x = self.pe(x)
        H_, W_ = x.shape[2:]
        x = self.blocks[0](Patch2Token(x), H_, W_)
        x = self.cpe(x, H_, W_)
        for block in self.blocks[1:]:
            x = block(x, H_, W_)
        return x
    
if __name__ == "__main__":
    torch.manual_seed(226)

    t = torch.rand((32, 64, 224, 224)).to("cuda")
    stage = TwinsStage(64, 128, 4, 4, 0, types=["lsa", "gsa"]).to("cuda")
    print(stage(t).size())
    stage = TwinsStage(64, 128, 5, 2, 2, types=["gsa", "gsa"]).to("cuda")
    print(stage(t).size())


