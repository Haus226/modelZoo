import torch
import math
from torch import nn

class EMA:
    def __init__(self, input_channels, k, momentum, num_iter):
        self.mu = torch.Tensor(1, input_channels, k)
        self.mu.normal_(0, math.sqrt(2. / k))    # Init with Kaiming Norm.
        self.mu = self.mu / (1e-6 + self.mu.norm(dim=1, keepdim=True))

        self.conv1 = nn.Conv2d(input_channels, input_channels, 1)
        self.conv2 = nn.Sequential(
                nn.Conv2d(input_channels, input_channels, 1, bias=False),
                nn.BatchNorm2d(input_channels)
            )     
        self.num_iter = num_iter
        self.momentum = momentum
        
    def forward(self, x):
        idn = x.clone()
        x = self.conv1(x)
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)               # b * c * n
        mu = self.mu.repeat(b, 1, 1)        # b * c * k
        with torch.no_grad():
            for _ in range(self.num_iter):
                x_t = x.permute(0, 2, 1)    # b * n * c
                # Eq.11 and 12 The normalized exp dot product is just softmax function.
                z = (x_t @ mu).softmax(dim=2)     # b * n * k
                # Eq. 13
                z = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = x @ z       # b * c * k
                # Section 5.3
                mu = mu / (1e-6 + mu.norm(dim=1, keepdim=True))

        # Eq. 14
        z_t = z.permute(0, 2, 1)            # b * k * n
        x = mu @ z_t                # b * c * n
        x = x.view(b, c, h, w).relu()             

        # The second 1x1 conv
        x = self.conv2(x)
        x = (x + idn).relu()

        # Moving average, this line only works during a training loop
        self.mu = self.momentum * self.mu + mu.mean(dim=0, keepdim=True) * (1 - self.momentum)

        return x

if __name__ == "__main__":
    t = torch.rand(8, 64, 32, 32)
    ema = EMA(64, 12, 0.9, 3)
    print(ema.forward(t).size())