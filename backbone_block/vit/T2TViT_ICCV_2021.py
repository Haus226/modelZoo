import torch
from torch import nn
from utils import Token2Patch, Transformer, Attention, Mlp


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

class T2TAttention(nn.Module):
    def __init__(self, in_channels, token_channels, num_heads, qkv_bias=True, attn_drop=0, proj_drop=0):
        super(T2TAttention, self).__init__()
        self.num_heads = num_heads
        self.token_channels = token_channels
        head_channels = token_channels // num_heads
        self.scale =  head_channels ** -0.5
        self.qkv = nn.Linear(in_channels, token_channels * 3, bias=qkv_bias)

        self.drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(token_channels, token_channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.size()
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.token_channels // self.num_heads).permute(2, 0, 3, 1, 4)
        # (B, N, C_embed)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        attn = q @ k.transpose(-2, -1)
        attn = self.drop(attn.softmax(dim=-1))
        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, self.token_channels)
        x_attn = self.proj_drop(self.proj(x_attn))
        # In the official repository, the skip connection is achieved through adding x_attn with v since
        # the original x has different shape. However, in their implementation, the v is being squeezed at second dimension (heads dimension.
        # And this can only be achieved when num_heads == 1  
        return x_attn + v.transpose(-2, -1).reshape(B, N, self.token_channels)
    
class T2TTransformer(Transformer):
    def __init__(self, in_channels, token_channels, num_heads=8, mlp_ratio=4, act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                qkv_bias=True, attn_drop=0, proj_drop=0, drop_path=0):
        super(T2TTransformer, self).__init__(in_channels, mlp_ratio, act_layer, norm_layer, proj_drop, drop_path, 0)
        self.norm_2 = nn.LayerNorm(token_channels)
        self.mlp = Mlp(token_channels, int(mlp_ratio * token_channels), act_layer=act_layer, norm_layer=norm_layer, drop=proj_drop)
        self.token_mixer = T2TAttention(in_channels, token_channels, num_heads, qkv_bias, attn_drop, proj_drop)
        delattr(self, "drop_path2")

    def forward(self, x):
        x = self.token_mixer(self.norm_1(x))
        x = x + self.drop_path1(self.mlp(self.norm_2(x)))
        return x


class T2TBlock(nn.Module):
    def __init__(self, in_channels, token_channels, embed_channels, depth,
                kernels_size, strides, 
                num_heads=8, mlp_ratio=4, 
                ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Unfold(kernels_size[0], stride=strides[0], padding=kernels_size[0] // 2))
        
        for idx in range(depth):
            self.blocks.append(
                T2TTransformer((token_channels if idx else in_channels) * kernels_size[idx] * kernels_size[idx],
                            token_channels, num_heads, mlp_ratio)
            )
            self.blocks.append(
                nn.Unfold(kernels_size[idx + 1], stride=strides[idx + 1], padding=kernels_size[idx + 1] // 2)
            )
        self.proj = nn.Linear(token_channels * kernels_size[-1] * kernels_size[-1], embed_channels)

    def forward(self, x):
        # (B, C, H, W) ---> (B, C * k * k, N) ---> (B, N, C * k * k)
        x = self.blocks[0](x).transpose(-1, -2)
        for idx in range(1, len(self.blocks), 2):
            x = self.blocks[idx](x)
            x = Token2Patch(x, int(x.shape[1] ** 0.5), w=int(x.shape[1] ** 0.5))
            x = self.blocks[idx + 1](x).transpose(-1, -2)
        x = self.proj(x)
        return x


if __name__ == "__main__":
    torch.manual_seed(226)
    x = torch.rand((4, 24, 32, 32))
    t2t = T2TBlock(24, 16, 128, 2, [7, 3, 3], [4, 2, 2])
    x = t2t(x)
    B, N, C = x.size()
    # Concatenate class token and add position embedding
    pos = nn.Parameter(torch.rand(1, N + 1, C))
    cls_tokens = nn.Parameter(torch.rand(1, 1, C)).repeat(B, 1, 1)

    x = torch.cat([cls_tokens, x], dim=1) + pos

    vit = ViTBlock(C, 4)
    x = vit(x)
    
    print(x.size())
