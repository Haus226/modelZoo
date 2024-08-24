import torch
from torch import nn

class NLNet:
    # Embedded Gaussian
    def __init__(self, input_channels, output_channels):
        self.q = nn.Conv2d(input_channels, output_channels, kernel_size=1, padding="same")
        self.k = nn.Conv2d(input_channels, output_channels, kernel_size=1, padding="same")
        self.v = nn.Conv2d(input_channels, output_channels, kernel_size=1, padding="same")  
        self.conv = nn.Conv2d(output_channels, input_channels, kernel_size=1, padding="same")

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w
        # theta, phi, g ---> q, k, v
        q_x = self.q(x)
        k_x = self.k(x)
        v_x = self.v(x)
        qk_x = q_x.view(b, -1, n).permute(0, 2, 1) @ k_x.view(b, -1, n)

        qk_x = qk_x.softmax(dim=-1)
        qkv_x = qk_x @ v_x.view(b, -1, n).permute(0, 2, 1)
        att = self.conv(qkv_x.view(b, -1, h, w))
        return x + att
    
if __name__ == "__main__":
    t = torch.rand(5, 1024, 64, 64)
    m = NLNet(1024, 512)
    print(m.forward(t).size())