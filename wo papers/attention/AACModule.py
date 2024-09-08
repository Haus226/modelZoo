'''
Refernces:
https://github.com/leaderj1001/Attention-Augmented-Conv2d/tree/master
http://arxiv.org/abs/1904.09925

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



class AAConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, nh, shape=0, relative=False, stride=1):
        super(AAConv, self).__init__()
        self.dk, self.dv, self.nh = dk, dv, nh
        self.shape = shape
        self.relative = relative
        padding = (kernel_size - 1) // 2

        assert self.nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.conv = nn.Conv2d(in_channels, out_channels - dv, kernel_size, stride=stride, padding=padding)
        self.qkv = nn.Conv2d(in_channels, 2 * self.dk + self.dv, kernel_size=1, stride=stride)
        self.attn = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)
        self.split_head = lambda x: rearrange(x, "B (N D) H W -> B N D H W", N=self.nh)

        if self.relative:
            self.key_rel_w = nn.Parameter(torch.randn((2 * self.shape - 1, dk // nh), requires_grad=True))
            self.key_rel_h = nn.Parameter(torch.randn((2 * self.shape - 1, dk // nh), requires_grad=True))

    def forward(self, x):
        conv_out = self.conv(x)

        flat_q, flat_k, flat_v, q = self.compute_flat_qkv(x, self.dk, self.dv, self.nh)
        # (B, N, H * W, D_k) @ (B, N, D_k, H * W) ---> (B, N, H * W, H * W) 
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits + w_rel_logits
        weights = logits.softmax(-1)
        # (B, N, H * W, H * W) @ (B, N, H * W, D_v) ---> (B, N, H * W, D_v)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = attn_out.view(conv_out.size(0), self.dv, *conv_out.size()[2:])
        attn_out = self.attn(attn_out)
        return torch.cat((conv_out, attn_out), dim=1)

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)

        # (B, N, C, H, W)
        q, k, v = self.split_head(q), self.split_head(k), self.split_head(v)

        # Dimension of key for each heads
        dkh = dk // Nh
        # Scaling factor
        q = q * dkh ** -0.5

        # (B, N, C, H * W)
        flat_q, flat_k, flat_v = q.flatten(start_dim=-2), k.flatten(start_dim=-2), v.flatten(start_dim=-2)
        return flat_q, flat_k, flat_v, q 

    def relative_logits(self, q):
        # The logits are shared across head (collapse head and one of the spatial dimension and repeat)
        B, _, _, H, W = q.size()

        rel_logits_w = torch.einsum("BNXYD, ZD->BNXYZ", rearrange(q, "B N D H W -> B N H W D"), self.key_rel_w)
        rel_logits_w = rel_logits_w.view(B, self.nh * H, W, 2 * W - 1)
        rel_logits_w = self.rel_to_abs(rel_logits_w)
        rel_logits_w = rel_logits_w.view(B, self.nh, H, W, W)
        # (B, N, H, W, W) ---> (B, N, H, 1, W, W)
        rel_logits_w = rel_logits_w.unsqueeze(3)
        rel_logits_w = rel_logits_w.repeat((1, 1, 1, H, 1, 1)) 
        # (B, N, H, H, W, W) ---> (B, N, H, W, H, W) ---> (B, N, H * W, H * W)
        rel_logits_w = torch.transpose(rel_logits_w, 3, 4).reshape(B, self.nh, H * W, H * W)
        
        rel_logits_h = torch.einsum("BNXYD, ZD->BNXYZ", rearrange(q, "B N D H W -> B N W H D"), self.key_rel_h)
        rel_logits_h = rel_logits_h.view(B, self.nh * W, H, 2 * H - 1)
        rel_logits_h = self.rel_to_abs(rel_logits_h)
        rel_logits_h = rel_logits_h.view(B, self.nh, W, H, H)
        # (B, N, W, H, H) ---> (B, N, W, 1, H, H)
        rel_logits_h = rel_logits_h.unsqueeze(3)
        rel_logits_h = rel_logits_h.repeat((1, 1, 1, W, 1, 1))
        # (B, N, W, W, H, H) ---> (B, N, H, W, H, W) ---> (B, N, H * W, H * W)
        rel_logits_h = rel_logits_h.permute((0, 1, 4, 2, 5, 3)).reshape(B, self.nh, H * W, H * W)

        return rel_logits_h, rel_logits_w

    # Magic part to extract absolute position embedding from relatvie position embedding
    def rel_to_abs(self, x):
        '''
        Example for relative position embedding: 
        The sentence, "This is a cat" has a length of 4, means that the relative position list
        has a length of 2 * 4 - 1 = 7
        Remember that every words in the sentence must use 0 as its position while 
        negative means before and positive means after 
              -3    -2    -1    0    1    2    3
        This                    0    1    2    3
        is                -1    0    1    2
        a           -2    -1    0    1
        cat   -3    -2    -1    0   

        Illustration of how the magic code extract the things we need
        Originally, the relative position embedding matrix (length = 3) has the shape below:
        -2    -1    0    1    2    
         @     @    0    1    2    
         @    -1    0    1    @    
        -2    -1    0    @    @    

        After col_pad:
        -2    -1    0    1    2    0   
         @     @    0    1    2    0
         @    -1    0    1    @    0
        -2    -1    0    @    @    0

        (Commas are just for better readability)
        After flatten (view):
        @     @    0    1    2    0,    @    -1    0    1    @    0,    -2    -1    0    @    @    0

        After row_pad
        @     @    0    1    2    0,    @    -1    0    1    @    0,    -2    -1    0    @    @    0,    0    0

        After reshape (view)
        @     @    0    1    2
        0     @    -1   0    1
        @     0    -2   -1   0
        @     @    0    0    0

        Now the right top corner is all the things we interested
        '''

        # (B, N, L, 2L - 1)
        B, N, L, _ = x.size()

        # (B, N, L, 2L)
        col_pad = torch.zeros((B, N, L, 1))
        x = torch.cat((x, col_pad), dim=3)

        # (B, N, 2 * L * L + L - 1)
        row_pad = torch.zeros((B, N, L - 1))
        x = x.view(B, N, 2 * L * L)
        x = torch.cat((x, row_pad), dim=2)

        x = x.view(B, N, L + 1, 2 * L - 1)
        x = x[:, :, :L, L - 1:]
        return x

if __name__ == "__main__":
    torch.manual_seed(226)
    t = torch.randn((16, 3, 32, 32))
    aac = AAConv(in_channels=3, out_channels=20, kernel_size=3, dk=40, dv=4, nh=4, 
                                    relative=True, stride=1, shape=32)
    print(aac.forward(t).size())