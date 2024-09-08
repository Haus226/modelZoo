'''
References:
http://arxiv.org/abs/1810.11579
'''

import torch
from torch import nn

class DAN:
    def __init__(self, input_channels, c_phi, c_theta):
        self.phi = nn.Conv2d(input_channels, c_phi, 1)
        self.theta = nn.Conv2d(input_channels, c_theta, 1)
        self.rho = nn.Conv2d(input_channels, c_theta, 1)
        self.conv = nn.Conv2d(c_phi, input_channels, 1)
        self.c_phi = c_phi
        self.c_theta = c_theta

    def forward(self, x):
        B, C, H, W = x.size()
        phi = self.phi(x).flatten(start_dim=-2)
        theta = self.theta(x).flatten(start_dim=-2).softmax(dim=-1)
        g = phi @ theta.permute(0, 2, 1)

        # According to the paper, the softmax function is applied to j dimension of v_ij
        # it should be flattened spatial dimension H * W
        rho = self.rho(x).flatten(start_dim=-2).softmax(dim=-1)
        z = g @ rho
        z = z.view(B, -1, H, W)
        return x + self.conv(z)

if __name__ == "__main__":
    t = torch.rand(32, 64, 21, 21)
    dan = DAN(64, 32, 16)
    print(dan.forward(t).size())
