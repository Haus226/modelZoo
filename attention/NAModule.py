import torch
from torch import nn
from torch.nn import functional as F

class NAM:
    def __init__(self, input_channels, height, width):
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.bn2 = nn.BatchNorm1d(height * width)

    def forward(self, x):
        b, c, h, w = x.size()   
        m_c = self.bn1(x)
        channel_att = self.bn1.weight.abs() / self.bn1.weight.abs().sum()
        m_c = m_c * channel_att.view(1, c, 1, 1)
        x = x * torch.sigmoid(m_c) 

        m_s = self.bn2(x.view(b, c, -1).permute(0, 2, 1))
        m_s = m_s.permute(0, 2, 1)
        spatial_att = self.bn2.weight.abs() / self.bn2.weight.abs().sum()
        m_s = m_s * spatial_att
        m_s = m_s.view(b, c, h, w)
        x = x * torch.sigmoid(m_s)

        return x
    
if __name__ == "__main__":
    t = torch.rand(5, 9, 24, 24)
    m = NAM(9, 24, 24)
    print(m.forward(t).size())
