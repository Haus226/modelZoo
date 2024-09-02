import torch

class SimAM:
    def __init__(self, gamma=1e-4):
        self.gamma = gamma

    def forward(self, x):
        _, _, H, W = x.size()
        # For unbiased variance
        n = H * W - 1
        d = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        v = d.sum(dim=(2, 3), keepdim=True) / n
        E_inv = d / (4 * (v + self.gamma)) + 0.5
        return x * E_inv.sigmoid()

if __name__ == "__main__":
    t = torch.rand(32, 64, 21, 21)
    simam = SimAM()
    print(simam.forward(t).size())
