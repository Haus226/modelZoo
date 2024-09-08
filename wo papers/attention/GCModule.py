import torch
from torch import nn

class GCBlock:
    def __init__(self, input_channels, r):
        self.k = nn.Conv2d(input_channels, 1, 1, bias=False)
        self.v1 = nn.Conv2d(input_channels, input_channels // r, 1, bias=False)
        self.v2 = nn.Conv2d(input_channels // r, input_channels, 1, bias=False)
        self.ln = nn.LayerNorm([input_channels // r, 1, 1])
        

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w
        k_x = self.k(x)
        k_x = k_x.view(b, 1, n)
        k_x = k_x.softmax(dim=-1)
        # (B, 1, C, HW) @ (B, 1, HW, 1) ---> Decide broadcastable only on batch dimension only
        # The output matrix size is (C, HW) @ (HW, 1) ---> (C, 1)
        # The output tensor size will be (B, 1, C, 1)

        att = x.view(b, c, -1).unsqueeze(1) @ k_x.unsqueeze(-1)
        att = att.view(b, c, 1, 1)

        att = self.v1.forward(att)
        att = self.ln.forward(att).relu()
        att = self.v2.forward(att)
        return x + att
    
if __name__ == "__main__":
    torch.manual_seed(226)
    C_in = 3
    S = 21
    t = torch.rand(3, C_in, S, S)
    g = GCBlock(C_in, 1)
    print(g.forward(t).size())