from torch import nn
import torch
from einops import rearrange
import torch.nn.functional as F
from timm.layers import DropPath
from inspect import signature

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, 
                kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                eps=1e-05, momentum=0.1, affine=True, norm=True,
                act=nn.ReLU):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels, eps, momentum, affine) if norm else nn.Identity()
        self.act = act() if act else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SEBlock(nn.Module):
    def __init__(self, input_channels:int, r:int):
        super(SEBlock, self).__init__()
        self.r = r
        self.fc1 = nn.Linear(input_channels, input_channels // r, bias=False)
        self.fc2 = nn.Linear(input_channels // r, input_channels, bias=False)
        # self.squeeze = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b, c, h, w = x.size()
        squeeze = torch.sum(x, (2, 3)) / (h * w)
        # With AdaptiveAvgPool2d -> self.squeeze(x) -> size = (b, c, 1, 1)
        # squeeze = self.squeeze(x).view(b, c)
        excitation = F.sigmoid(self.fc2(F.relu(self.fc1(squeeze)))).view(b, c, 1, 1)
        return x * excitation.expand_as(x)

class Mlp(nn.Module):
    def __init__(self, in_channels, mid_channels=None, out_channels=None, 
                act_layer=nn.GELU, norm_layer=None, 
                bias=True, drop=0, use_conv=False):
        super(Mlp, self).__init__()
        out_channels = out_channels or in_channels
        mid_channels = mid_channels or in_channels
        self.fc1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias) if use_conv else nn.Linear(in_channels, mid_channels, bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(mid_channels) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Conv2d(mid_channels, out_channels, 1, bias=bias) if use_conv else nn.Linear(mid_channels, out_channels, bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Attention(nn.Module):
    def __init__(self, channels, num_heads=32, qkv_bias=True, attn_drop=0, proj_drop=0):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_channels = channels // num_heads
        self.scale = head_channels ** -0.5
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)

        self.drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.size()

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.chunk(3, dim=0)
        q = q * self.scale

        attn = q @ k.transpose(-2, -1)
        attn = self.drop(attn.softmax(dim=-1))

        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_attn = self.proj_drop(self.proj(x))
        return x_attn

class Transformer(nn.Module):
    def __init__(self, channels, mlp_ratio=4,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm, proj_drop=0,
                drop_path=0, layer_scale=0):
        super(Transformer, self).__init__()
        self.token_mixer = None
        self.norm_1 = norm_layer(channels)
        self.norm_2 = norm_layer(channels)
        self.mlp = Mlp(channels, int(channels * mlp_ratio), act_layer=act_layer, norm_layer=norm_layer, drop=proj_drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(channels)) if layer_scale > 0 else 1.0
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(channels)) if layer_scale > 0 else 1.0

    def forward_features(self, x, layer, norm, drop, scale, H=None, W=None):
        params = signature(layer.forward).parameters
        res = x
        x = layer(norm(x), H, W) if 'H' in params and 'W' in params else layer(norm(x))
        return res + drop(scale * x)

    def forward(self, x, H=None, W=None):

        assert self.token_mixer is not None, "You may forgot to define the attention part in the transformer"
        x = self.forward_features(x, self.token_mixer, self.norm_1, self.drop_path1, self.gamma_1, H, W)
        x = self.forward_features(x, self.mlp, self.norm_2, self.drop_path2, self.gamma_2, H, W)
        return x

class GroupNorm(nn.GroupNorm):
    '''
    GroupNorm is equivalent to LayerNorm when num_groups = 1
    '''
    def __init__(self, num_channels, **kwargs):
        super(GroupNorm, self).__init__(1, num_channels, **kwargs)

class PatchEmbeddingV1(nn.Module):
    '''
    Patch embedding used for downsampling that accept input with shape (B, C, H, W) and
    return output with shape by considering different cases below:

    For odd patch_size, padding is performed where padding == patch_size // 2 or without padding, then
    With padding
    (B, C, H, W) ---> (B, C, floor((H - 1) / S + 1), floor((W - 1) / S + 1))
    Without padding
    (B, C, H, W) ---> (B, C, floor((H - K) / S + 1), floor((W - K) / S + 1))
    
    For even patch_size, padding is not performed and normally stride ==  patch_size, then
    (B, C, H, W) ---> (B, C, H // patch_size, W // patch_size)
    '''
    def __init__(self, in_channels, embed_channels, patch_size, stride, padding=0, norm=None):
        super(PatchEmbeddingV1, self).__init__()
        self.embed = nn.Conv2d(in_channels, embed_channels, patch_size, stride, padding)
        self.norm = norm(embed_channels) if norm is not None else nn.Identity()

    def forward(self, x):
        return self.norm(self.embed(x))
    
class PatchEmbeddingV2(nn.Module):
    '''
    Patch embedding for downsampling that accept input with shape (B, N, C) and
    reshape it into (B, C, H, W) where N must equals to H * W.

    For odd patch_size, padding is performed where padding == patch_size // 2 or without padding, then
    With padding
    (B, C, H, W) ---> (B, floor((H - 1) / S + 1) * floor((W - 1) / S + 1), C)
    Without padding
    (B, C, H, W) ---> (B, floor((H - K) / S + 1) * floor((W - K) / S + 1), C)

    For even patch_size, padding is not performed and normally stride ==  patch_size, then
    (B, C, H, W) ---> (B, H // patch_size * W // patch_size, C)

    If norm will be LayerNorm, use nn.GroupNorm where num_groups == 1
    '''

    def __init__(self, in_channels, embed_channels, patch_size, stride, padding=0, norm=GroupNorm):
        super(PatchEmbeddingV2, self).__init__()
        self.embed = nn.Conv2d(in_channels, embed_channels, patch_size, stride, padding)
        self.norm = norm(embed_channels) if norm is not None else nn.Identity()

    def forward(self, x, H, W):
        B, N, C = x.size()
        assert N == H * W
        return rearrange(self.norm(self.embed(x.reshape(B, H, W, C).permute(0, 3, 1, 2))), "b c h w -> b (h w) c")

class CPE(nn.Module):
    '''
    Title: CONDITIONAL POSITIONAL ENCODINGS FOR VISION TRANSFORMERS\n
    References: https://arxiv.org/abs/2102.10882\n
    Used in Twins
    '''
    def __init__(self, in_channels, embed_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super(CPE, self).__init__()
        self.conv = nn.Conv2d(in_channels, embed_channels, kernel_size, stride, padding, groups=groups)
        self.stride = stride

    def forward(self, x, H, W):
        assert x.size(1) == H * W
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.conv(x) + x if self.stride == 1 else self.conv(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return x

def rel_pos_idx(height, width):
    coords = torch.meshgrid((torch.arange(height), torch.arange(width)), indexing="ij")
    coords = torch.flatten(torch.stack(coords), 1)
    # Compute the 2D relative position for every coordinates
    relative_coords = coords[:, :, None] - coords[:, None, :]

    # Since the bias is zero-indexed, the relative coordinates are add with the offsets.
    # The value is between (0, H) and (0, W), including H and W
    relative_coords[0] += height - 1
    relative_coords[1] += width - 1

    # Since the bias is 1D with shape (2H - 1) * (2W - 1), multiply the first dimension with the
    # number of elements in the second dimension which is the stride of first dimension.
    relative_coords[0] *= 2 * width - 1

    # Add the first and the second dimension which is actually row and column indices to get the indices
    # of flatten bias.
    relative_indices = relative_coords.sum(0)

    # The output has shape (H * W * H * W)
    return relative_indices.flatten()

def WP(x, window_height, window_width):
    '''
    Window partition that accept input with shape (B, C, H, W) 
    and return it as shape (B * H // WIN_H * W // WIN_W, WIN_H * WIN_W, C)

    Procedures:
    (B, C, H, W) ---> (B, C, H // WIN_H, WIN_H, W // WIN_W, WIN_W) ---> (B * H // WIN_H * W // WIN_W, WIN_H * WIN_W, C)
    '''
    return rearrange(x, "b c (h wh) (w ww) -> (b h w) (wh ww) c", wh=window_height, ww=window_width)

def RWP(x, h, w, window_height, window_width):
    '''
    Reverse window partition
    (B * H // WIN_H * W // WIN_W, WIN_H * WIN_W, C) ---> (B, C, H, W)
    '''
    return rearrange(x, "(b h w) (wh ww) c -> b c (h wh) (w ww)", h=h // window_height, w=w // window_width, wh=window_height, ww=window_width)

def GP(x, grid_height, grid_width):
    '''
    Grid partition that accept input with shape (B, C, H, W) 
    and return it as shape (B * H // GRID_H * W // GRID_W, GRID_H * GRID_W, C)

    Procedures:
    (B, C, H, W) ---> (B, C, GRID_H, H // GRID_H, GRID_W, W // GRID_W) ---> (B * H // GRID_H * W // GRID_W, GRID_H * GRID_W, C)
    '''
    return rearrange(x, "b c (gh h) (gw w) -> (b h w) (gh gw) c", gh=grid_height, gw=grid_width)

def RGP(x, h, w, grid_height, grid_width):
    '''
    Reverse grid partition
    (B * H // GRID_H * W // GRID_W, GRID_H * GRID_W, C) ---> (B, C, H, W)
    '''
    return rearrange(x, "(b h w) (gh gw) c -> b c (gh h) (gw w)", h=h // grid_height, w=w // grid_width, gh=grid_height, gw=grid_width)

def Token2Patch(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

def Patch2Token(x):
    return rearrange(x, "b c h w -> b (h w) c")

def MergeHeads(x):
    return rearrange(x, "b h n c -> b n (h c)")

def SplitHeads(x):
    return rearrange()