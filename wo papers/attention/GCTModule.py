from torch import nn
import torch

class GCT:
    def __init__(self, input_channels, alpha=None, gamma=None, beta=None, eps=1e-5):
        if alpha is None:
            self.alpha = nn.Parameter(torch.ones((1, input_channels, 1, 1)))
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha).view(1, input_channels, 1, 1))
        if gamma is None:
            self.gamma = nn.Parameter(torch.zeros((1, input_channels, 1, 1)))
        else:
            self.gamma = nn.Parameter(torch.tensor(gamma).view(1, input_channels, 1, 1))
        if beta is None:
            self.beta = nn.Parameter(torch.zeros((1, input_channels, 1, 1)))
        else:
            self.beta = nn.Parameter(torch.tensor(beta).view(1, input_channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        embedding = self.alpha * (x.pow(2).sum(dim=(2, 3), keepdim=True) + self.eps).pow(0.5)
        norm = self.gamma / (embedding.mean(dim=1, keepdim=True) + self.eps).pow(0.5)
        gate = 1 + torch.tanh(embedding * norm + self.beta)
        return x * gate

if __name__ == "__main__":
    t = torch.rand(32, 128, 21, 21)
    gct = GCT(128)
    print(gct.forward(t).size())