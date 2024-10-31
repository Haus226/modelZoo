import torch
from torch import nn
from utils import PatchEmbeddingV1, WP, MergeHeads, Attention, rel_pos_idx, Transformer, RWP, Patch2Token, Token2Patch
from timm.layers import DropPath
from einops import rearrange

class RelAttention(Attention):
    def __init__(self, channels, num_heads=32, region_size=[7, 7], qkv_bias=True, attn_drop=0, proj_drop=0):
        super().__init__(channels, num_heads, qkv_bias, attn_drop, proj_drop)
        self.region_size = (region_size, region_size) if isinstance(region_size, int) else region_size
        self.rel_pos_bias = nn.Parameter(
            torch.rand(((2 * self.region_size[0] - 1) * (2 * self.region_size[1] - 1), num_heads))
        )
        self.rel_pos_idx = rel_pos_idx(self.region_size[0], self.region_size[1])
        self.num_heads = num_heads

    def forward(self, x, type="rsa"):
        B, N, C = x.size()

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        attn = q @ k.transpose(-2, -1)
        if type == "lsa":
            rel_pos_bias = torch.zeros((self.region_size[0] * self.region_size[1] + 1, self.region_size[0] * self.region_size[1] + 1, 
                                        self.num_heads), device=attn.device)
            rel_pos_bias[1:, 1:] = self.rel_pos_bias[self.rel_pos_idx].view(
                self.region_size[0] * self.region_size[1], self.region_size[0] * self.region_size[1], -1)
            rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()  
            attn = attn + rel_pos_bias.unsqueeze(0)
        
        attn = self.drop(attn.softmax(dim=-1))
        x_attn = MergeHeads(attn @ v)
        x_attn = self.proj_drop(self.proj(x_attn))
        return x_attn


class RegionViTBlock(Transformer):
    def __init__(self, channels, region_size, num_heads, qkv_bias=True, mlp_ratio=4, 
                attn_drop=0, proj_drop=0, drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(RegionViTBlock, self).__init__(channels, mlp_ratio, act_layer, norm_layer, proj_drop, drop_path)
        self.token_mixer = RelAttention(channels, num_heads, region_size, qkv_bias, attn_drop, proj_drop)
        self.norm_3 = nn.LayerNorm(channels)
        self.drop_path3 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.region_size = region_size

    def forward(self, local_tokens, region_tokens, local_H, local_W):
        n = region_tokens.size(1)
        '''
        local_tokens: (B, C, H // local_size, W // local_size)
        region_tokens: (B, H // local_size * region_size * W // local_size * region_size, C)
        where regional_size is multiple of local_size
        '''
        region_tokens = region_tokens + self.drop_path1(self.norm_1(self.token_mixer(region_tokens)))

        # (B, C, H // LOCAL_H, W // LOCAL_W) ---> 
        # (B * H // LOCAL_H // REGION_H * W // WIN_W // REGION_W, REGION_H * REGION_W, C)
        local_tokens = WP(local_tokens, self.region_size, self.region_size)
        region_tokens = rearrange(region_tokens, "b n c -> (b n) 1 c")
        concated = torch.cat([region_tokens, local_tokens], dim=1)

        concated = concated + self.drop_path2(self.norm_2(self.token_mixer(concated, "lsa")))
        concated = concated + self.drop_path3(self.norm_3(self.mlp(concated)))

        region_tokens, local_tokens = concated[:, :1], concated[:, 1:]
        region_tokens = rearrange(region_tokens, "(b n) 1 c -> b n c", n=n)
        local_tokens = RWP(local_tokens, local_H, local_W, self.region_size, self.region_size)
        return local_tokens, region_tokens
    
class RegionViTStage(nn.Module):
    def __init__(self, channels, embed_channels, num_blocks=2, num_heads=8, qkv_bias=True, 
                attn_drop=0, proj_drop=0, drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                local_size=4, regional_size=7,
                mlp_ratio=4, ):
        super(RegionViTStage, self).__init__()
        self.region_pe = PatchEmbeddingV1(channels, embed_channels, local_size * regional_size, local_size * regional_size, 
                                          (local_size * regional_size) // 2 if (local_size * regional_size) % 2 else 0, method="linear")
        self.local_pe = PatchEmbeddingV1(channels, embed_channels, local_size, local_size, 
                                         local_size // 2 if local_size % 2 else 0)
        self.blocks = nn.ModuleList([
            RegionViTBlock(embed_channels, regional_size, num_heads, qkv_bias, mlp_ratio, attn_drop, proj_drop, drop_path, act_layer, norm_layer)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        region_tokens = self.region_pe(x)
        local_tokens = self.local_pe(x)
        H_, W_ = local_tokens.shape[2:]
        # local_tokens will have 4 dimensions since we need to partition it into windows in
        # following layers.
        # region_tokens will have 3 dimensions
        local_tokens, region_tokens = self.blocks[0](local_tokens, 
                                                    Patch2Token(region_tokens), 
                                                    H_, W_)
        for block in self.blocks[1:]:
            local_tokens, region_tokens = block(local_tokens, region_tokens, H_, W_)
        return local_tokens, region_tokens
    
if __name__ == "__main__":
    torch.manual_seed(226)
    t = torch.rand((2, 2, 224, 224)).to("cuda")
    stage = RegionViTStage(2, 4, 2, 4).to("cuda")
    local_tokens, region_tokens = stage(t)
    print(local_tokens.size(), region_tokens.size())
    t = torch.rand((32, 64, 120, 120)).to("cuda")
    stage = RegionViTStage(64, 128, 2, 2, local_size=5, regional_size=6).to("cuda")
    local_tokens, region_tokens = stage(t)
    print(local_tokens.size(), region_tokens.size())