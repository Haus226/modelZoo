import torch
from torch import nn

class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias=True):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

class SKM:
    def __init__(self, input_channels: int, branches: int = 2, 
                groups: int = 32, stride: int = 1, 
                r: int = 16, min_internal_channels: int = 32):
        
        internal_channels = max(input_channels // r, min_internal_channels)
        self.M = branches
        self.input_channels = input_channels
        
        # Split
        self.multi_branch_convs = nn.ModuleList([])
        for i in range(self.M):
            self.multi_branch_convs.append(nn.Sequential(
                ConvBN(input_channels, input_channels, kernel_size=3, stride=stride, padding="same", dilation=1 + i, groups=groups, bias=False),
                nn.ReLU(inplace=True)
            ))
        
        # Fuse
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(ConvBN(input_channels, internal_channels, kernel_size=1, stride=1, bias=False),
                                nn.ReLU(inplace=True))
        
        # Select
        self.multi_branch_fcs = nn.ModuleList([])
        for i in range(self.M):
            self.multi_branch_fcs.append(nn.Conv2d(internal_channels, input_channels, kernel_size=1, stride=1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, h, w = x.size()
        
        feats = [conv(x) for conv in self.multi_branch_convs]   # (b, c, h, w) * m
        feats = torch.cat(feats, dim=1).view(b, self.M, c, h, w)    # (b, m, c, h, w)
        
        feats_U = torch.sum(feats, dim=1)   # (b, c, h, w)
        feats_s = self.gap(feats_U) # (b, c, 1, 1)
        feats_z = self.fc(feats_s)  # (b, c / r, 1, 1)

        att = [fc(feats_z) for fc in self.multi_branch_fcs] # (b, c, 1, 1) * m
        att = torch.cat(att, dim=1) # (b, m * c, 1, 1)
        att = att.view(b, self.M, c, 1, 1)  # (b, m, c, 1, 1)
        att = self.softmax(att)

        return torch.sum(att * feats, dim=1)    # (b, c, h, w)
    
if __name__ == "__main__":
    t = torch.rand(32, 64, 21, 21)
    s = SKM(64)
    print(s.forward(t).size())