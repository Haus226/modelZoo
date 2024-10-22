import torch
from torch import nn
from utils import Transformer, WP, RWP, Token2Patch, Patch2Token, MergeHeads
from einops import rearrange
from torchvision.ops import DeformConv2d

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
    

class DWConv(nn.Module):
    def __init__(self, channels):
        super(DWConv, self).__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

    def forward(self, x, H, W):
        '''
        x should be in shape of (B, N, C)
        '''
        return Patch2Token(self.dw(Token2Patch(x, H, W)))

class Mlp(nn.Module):
    def __init__(self, in_channels, mid_channels=None, out_channels=None, 
                act_layer=nn.GELU, norm_layer=None, 
                bias=True, drop=0, use_conv=False):
        super(Mlp, self).__init__()
        out_channels = out_channels or in_channels
        mid_channels = mid_channels or in_channels
        self.pre_norm = nn.LayerNorm(in_channels)
        self.fc1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias) if use_conv else nn.Linear(in_channels, mid_channels, bias)
        self.act1 = nn.ReLU()
        self.dw = DWConv(mid_channels)
        self.act2 = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(mid_channels) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Conv2d(mid_channels, out_channels, 1, bias=bias) if use_conv else nn.Linear(mid_channels, out_channels, bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.pre_norm(x)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dw(x, H, W)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class HiLoAttention(nn.Module):
    def __init__(self, channels, alpha, num_heads=32, window_size=[2, 2], qkv_bias=True, attn_drop=0, proj_drop=0):
        super(HiLoAttention, self).__init__()
        head_channels = channels // num_heads
        self.scale = head_channels ** -0.5

        self.low_heads = int(alpha * num_heads)
        self.low_channels = head_channels * self.low_heads

        self.high_heads = num_heads - self.low_heads
        self.high_channels = head_channels * self.high_heads
        self.window_size = window_size

        self.h_qkv = nn.Linear(channels, self.high_channels * 3, bias=qkv_bias)
        self.h_proj = nn.Linear(self.high_channels, self.high_channels)

        self.avg = nn.AvgPool2d(window_size, window_size)
        self.l_q = nn.Linear(channels, self.low_channels, bias=qkv_bias)
        self.l_kv = nn.Linear(channels, self.low_channels * 2, bias=qkv_bias)
        self.l_proj = nn.Linear(self.low_channels, self.low_channels)

    def hifi(self, x):
        _, _, H, W = x.size()
        x = WP(x, self.window_size[0], self.window_size[1])
        B_, N, _ = x.size()
        qkv = self.h_qkv(x).reshape(B_, N, 3, self.high_heads, self.high_channels // self.high_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = MergeHeads(attn @ v)
        attn = self.h_proj(attn)
        attn = RWP(attn, H, W, self.window_size[0], self.window_size[1])
        return attn
    
    def lifi(self, x):
        B, _, H, W = x.size()
        q = self.l_q(Patch2Token(x)).reshape(B, -1, self.low_heads, self.low_channels // self.low_heads).permute(0, 2, 1, 3)

        x = Patch2Token(self.avg(x))
        kv = self.l_kv(x).reshape(B, -1, 2, self.low_heads, self.low_channels // self.low_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = MergeHeads(attn @ v)
        attn = Token2Patch(self.l_proj(attn), H, W)
        return attn

    def forward(self, x, H, W):
        x = Token2Patch(x, H, W)
        x_high = self.hifi(x)
        x_low = self.lifi(x)
        x = torch.cat([x_high, x_low], dim=1)
        return Patch2Token(x)
    
class LITv2Block(nn.Module):
    def __init__(self, channels,alpha, num_heads, window_size, qkv_bias, mlp_ratio=4,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                attn_drop=0, drop=0, drop_path=0, type="mlp"):
        super(LITv2Block, self).__init__()
        if type == "mlp":
            self.block = Mlp(channels, int(mlp_ratio * channels), act_layer=act_layer, norm_layer=norm_layer, drop=drop)
        else:
            self.block = Transformer(channels, int(mlp_ratio * channels), act_layer, norm_layer, 
                                    drop, drop_path)
            self.block.token_mixer = HiLoAttention(channels, alpha, num_heads, window_size, qkv_bias, attn_drop, drop)
    
    def forward(self, x, H, W):
        return self.block(x, H, W)

    
class LITv2Stage(nn.Module):
    def __init__(self, in_channels, embed_channels, patch_size, stride, padding, dilation=1, modulated=True,
                alpha=0.5, window_size=[2, 2],
                types=["mlp"] * 4,
                num_heads=8, mlp_ratio=4, qkv_bias=True, 
                proj_drop=0, attn_drop=0, drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm
                ):
        super(LITv2Stage, self).__init__()
        self.dtm = DTM(in_channels, embed_channels, patch_size, stride, padding, dilation, modulated)
        self.blocks = nn.ModuleList([
            LITv2Block(embed_channels, alpha, num_heads, window_size, qkv_bias, mlp_ratio, 
                    act_layer, norm_layer, proj_drop, attn_drop, drop_path, types[idx]) for idx in range(len(types))]
        )

    def forward(self, x):
        x = self.dtm(x)
        H_, W_ = x.shape[2:]
        x = self.blocks[0](Patch2Token(x), H_, W_)
        for block in self.blocks[1:]:
            x = block(x, H_, W_)
        return x
    
if __name__ == "__main__":
    torch.manual_seed(226)
    t = torch.rand((32, 64, 24, 24)).to("cuda")
    stage = LITv2Stage(64, 128, 4, 4, 0, types=["mlp", "mlp"]).to("cuda")
    print(stage(t).size())
    stage = LITv2Stage(64, 128, 5, 2, 2, types=["t", "t"]).to("cuda")
    print(stage(t).size())
