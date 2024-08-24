import torch
from torch import nn

class Attention:
    def __init__(self, permutation) -> None:
        self.permutation = permutation
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding="same")
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        x_rot = x.permute(self.permutation)
        x_t = torch.cat([x_rot.max(dim=1, keepdim=True)[0], x_rot.mean(dim=1, keepdim=True)], dim=1)
        x_t = self.conv(x_t)
        x_t = self.bn(x_t)
        x_reverse = x_t.sigmoid()
        return (x_rot * x_reverse).permute(self.permutation)

class TAM:
    def __init__(self):
        self.HW_att = Attention((0, 1, 2, 3))
        self.CW_att = Attention((0, 2, 1, 3))
        self.HC_att = Attention((0, 3, 2, 1))

    def forward(self, x):
        hw_att = self.HW_att.forward(x)
        cw_att = self.CW_att.forward(x)
        hc_att = self.HC_att.forward(x)
        return (hw_att + cw_att + hc_att) / 3

if __name__ == "__main__":
    t = torch.rand(5, 12, 64, 32)
    m = TAM()
    print(m.forward(t).size())