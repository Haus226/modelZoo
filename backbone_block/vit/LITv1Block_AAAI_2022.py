'''
Title
Less is More: Pay Less Attention in Vision Transformers

Refereces
http://arxiv.org/abs/2105.14217
'''



import torch
from torch import nn
from torchvision.ops import DeformConv2d
from utils import Mlp, Attention, Transformer, Patch2Token, rel_pos_idx, MergeHeads
from math import floor

class RelAttention(Attention):
    def __init__(self, channels, num_heads=32, patch_size=[1, 1], qkv_bias=True, attn_drop=0, proj_drop=0):
        super().__init__(channels, num_heads, qkv_bias, attn_drop, proj_drop)
        self.patch_size = patch_size
        self.rel_pos_bias = nn.Parameter(
            torch.rand(((2 * patch_size[0] - 1) * (2 * patch_size[1] - 1), num_heads))
        )
        self.rel_pos_idx = rel_pos_idx(patch_size[0], patch_size[1])

    def forward(self, x):
        B, N, C = x.size()

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        attn = q @ k.transpose(-2, -1)
        rel_pos_bias = self.rel_pos_bias[self.rel_pos_idx].view(
            self.patch_size[0] * self.patch_size[1], self.patch_size[0] * self.patch_size[1], -1)
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()  
        attn = attn + rel_pos_bias.unsqueeze(0)

        attn = self.drop(attn.softmax(dim=-1))
        x_attn = MergeHeads(attn @ v)
        x_attn = self.proj_drop(self.proj(x_attn))
        return x_attn

    

class DTM(nn.Module):
    def __init__(self, in_channels, embed_channels, patch_size=3, 
                stride=1, padding=0, dilation=1, modulated=True):
        super(DTM, self).__init__()
        self.modulated = modulated
        # Modulation and Offset Convolution, can be combined into one layer and 
        # chunk into 2 parts.
        self.m_conv = nn.Conv2d(in_channels, patch_size * patch_size, patch_size, stride, 
                                padding, dilation)
        self.p_conv = nn.Conv2d(in_channels, 2 * patch_size * patch_size, patch_size, stride, padding, dilation)
        self.deform_conv = DeformConv2d(in_channels, embed_channels, patch_size, stride, padding, dilation)
        self.norm = nn.BatchNorm2d(embed_channels)
        self.act = nn.GELU()

    def forward(self, x):
        offset = self.p_conv(x)
        mask = self.m_conv(x).sigmoid() if self.modulated else None
        return self.act(self.norm(self.deform_conv(x, offset, mask)))
    
class LITv1Block(nn.Module):
    def __init__(self, channels, num_heads, patch_size, qkv_bias, mlp_ratio=4,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                attn_drop=0, drop=0, drop_path=0, type="mlp"):
        super(LITv1Block, self).__init__()
        if type == "mlp":
            self.block = nn.Sequential(
                nn.LayerNorm(channels),
                Mlp(channels, int(mlp_ratio * channels), act_layer=act_layer, norm_layer=norm_layer, drop=drop),
            )
        else:
            self.block = Transformer(channels, int(mlp_ratio * channels), act_layer, norm_layer, 
                                    drop, drop_path)
            self.block.token_mixer = RelAttention(channels, num_heads, patch_size, qkv_bias, attn_drop, drop)
    
    def forward(self, x):
        return self.block(x)

class LITv1Stage(nn.Module):
    def __init__(self, in_channels, embed_channels, input_spatial, patch_size, stride, padding, dilation=1, modulated=True,
                types=["mlp"] * 4,
                num_heads=8, mlp_ratio=4, qkv_bias=True, 
                proj_drop=0, attn_drop=0, drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm
                ):
        super(LITv1Stage, self).__init__()
        self.dtm = DTM(in_channels, embed_channels, patch_size, stride, padding, dilation, modulated)
        H_ = floor((input_spatial[0] - patch_size + 2 * padding) / stride + 1)
        W_ = floor((input_spatial[1] - patch_size + 2 * padding) / stride + 1)

        self.blocks = nn.ModuleList([
            LITv1Block(embed_channels, num_heads, [H_, W_], qkv_bias, mlp_ratio, 
                    act_layer, norm_layer, proj_drop, attn_drop, drop_path, types[idx]) for idx in range(len(types))]
        )

    def forward(self, x):
        x = self.dtm(x)
        x = self.blocks[0](Patch2Token(x))
        for block in self.blocks[1:]:
            x = block(x)
        return x
    
if __name__ == "__main__":
    torch.manual_seed(226)
    t = torch.rand((32, 64, 32, 32))
    dtm = DTM(64, 32, 3, 2)
    print(dtm(t).size())

    torch.manual_seed(226)

    t = torch.rand((32, 64, 21, 21)).to("cuda")
    stage = LITv1Stage(64, 128, [21, 21], 4, 4, 0, types=["mlp", "mlp"]).to("cuda")
    print(stage(t).size())
    stage = LITv1Stage(64, 128, [21, 21], 5, 2, 2, types=["t", "t"]).to("cuda")
    print(stage(t).size())