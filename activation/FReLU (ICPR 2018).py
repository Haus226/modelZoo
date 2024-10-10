import torch
from torch import nn

class FReLU(nn.ReLU):
    def __init__(self, channels, inplace=False):
        super(FReLU, self).__init__(inplace)
        self.bias = nn.Parameter(torch.rand(1, channels, 1, 1))

    def forward(self, x):
        return super(FReLU, self).forward(x) + self.bias
    
if __name__ == "__main__":
    torch.manual_seed(226)
    t = torch.rand((32, 64, 21, 21))
    criterion = nn.MSELoss()
    model = nn.Sequential(
        nn.Conv2d(64, 128, 3, padding=1),
        FReLU(128)
    )
    loss = criterion(model(t), torch.rand(32, 128, 21, 21))
    loss.backward()
    
