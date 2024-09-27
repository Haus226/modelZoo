'''
Title
Going Deeper with Convolutions

References
http://arxiv.org/abs/1409.4842
'''



import torch
from torch import nn
from utils import ConvBNReLU

class InceptionV1Block(nn.Module):
    def __init__(self, in_channels, out1x1, 
                reduce3x3, out3x3,
                reduce5x5, out5x5, 
                out_proj, aux=False):
        super(InceptionV1Block, self).__init__()
        self.branch_1x1 = ConvBNReLU(in_channels, out1x1, 1)
        self.branch_3x3 = nn.Sequential(
            ConvBNReLU(in_channels, reduce3x3, 1),
            ConvBNReLU(reduce3x3, out3x3, 3, padding=1)
        )
        self.branch_5x5 = nn.Sequential(
            ConvBNReLU(in_channels, reduce5x5, 1),
            ConvBNReLU(reduce5x5, out5x5, 5, padding=2)
        )
        self.branch_proj = nn.Sequential(
            nn.MaxPool2d(3, 1, padding=1),
            ConvBNReLU(in_channels, out_proj, 1)
        )

        if aux:
            self.aux = True
            self.branch_aux = nn.Sequential(
                nn.AvgPool2d(5, 3),
                ConvBNReLU(in_channels, 128, 1),
                # From the paper, this should be nn.Linear(4 * 4 * 128),
                # the spatial dimension after Average Pooling should be 4 until this stage.
                nn.Flatten(),
                nn.LazyLinear(1024),
                nn.Dropout(0.7),
                nn.Linear(1024, 1000),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        return torch.cat([self.branch_1x1(x), self.branch_3x3(x), self.branch_5x5(x), self.branch_proj(x)], dim=1), self.branch_aux(x) if self.aux else x 

if __name__ == "__main__":
    torch.manual_seed(226)
    t = torch.rand(32, 64, 21, 21)
    inception = InceptionV1Block(64, 192, 96, 208, 16, 48, 64, True)
    out, aux = inception(t)
    print(out.size(), aux.size())