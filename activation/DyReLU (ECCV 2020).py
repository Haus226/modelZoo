'''
Title
Dynamic ReLU

References
http://arxiv.org/abs/2003.10027
'''



import torch
from torch import nn

class TemperatueSoftmax(nn.Module):
    def __init__(self, dim, temp=10):
        super(TemperatueSoftmax, self).__init__()
        self.softmax = nn.Softmax(dim)
        self.temp = temp
    
    def forward(self, x):
        x = x / self.temp
        return self.softmax(x)


class DyReLU(nn.Module):
    def __init__(self, channels, k, r, gamma, temp, 
                lambdas, init_vals, 
                type:str):
        super(DyReLU, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.channel_block = nn.Sequential(
            nn.Linear(channels, channels // r),
            nn.ReLU(),
            nn.Linear(channels // r, 
                    2 * k * channels if type.lower() in ["b", "c"] else 2 * k),
            nn.Sigmoid()
        ) 
        if type.lower() == "c":
            self.spatial_block = nn.Sequential(
                nn.Conv2d(channels, 1, 1),
                nn.Flatten(),
                TemperatueSoftmax(dim=1)
            )
        self.k = k
        self.gamma = gamma
        self.temp = temp
        self.type = type
        self.lambdas = torch.Tensor(lambdas)
        self.init_vals = torch.Tensor(init_vals)
    
    def forward(self, x):
        B, C, H, W = x.size()
        C = C if self.type.lower() in ["b", "c"] else 1
        H = H if self.type.lower() == "c" else 1
        H = W if self.type.lower() == "c" else 1

        c_coef = self.gap(x).squeeze()
        c_coef = self.channel_block(c_coef)
        # The normalized a's and b's in equation 2
        c_coef = 2 * c_coef - 1
        c_coef = c_coef.view(B, C, 2 * self.k) * self.lambdas + self.init_vals

        s_coef = 1
        if self.type.lower() == "c":
            s_coef = self.spatial_block(x)
            s_coef = torch.minimum(self.gamma * s_coef, torch.ones_like(s_coef))
            s_coef = s_coef.view(B, 1, H, W, 1)
        c_coef = c_coef.view(B, C, 1, 1, 2 * self.k) * s_coef
        
        y = x.unsqueeze(-1) * c_coef[:, :, :, :, :self.k] + c_coef[:, :, :, :, self.k:]

        # if self.type.lower() == "a":
        #     coef = coef * self.lambdas + self.init_vals
        #     y = x.unsqueeze(-1) * coef[:, :self.k].view(B, 1, 1, 1, self.k) + coef[:, self.k:].view(B, 1, 1, 1, self.k)
        # elif self.type.lower() == "b":
        #     coef = coef.view(B, C, 2 * self.k) * self.lambdas + self.init_vals
        #     y = x.unsqueeze(-1) * coef[:, :, :self.k].view(B, C, 1, 1, self.k) + coef[:, :, self.k:].view(B, C, 1, 1, self.k)
        return y.max(dim=-1)[0]

if __name__ == "__main__":
    torch.manual_seed(226)
    t = torch.rand((32, 64, 21, 21))
    dya = DyReLU(64, 2, 4, 21 ** 2 / 3, 10, lambdas=[1.09124] * 2  + [0.5] * 2, init_vals=[1] + [1.09124] * 3, type="a")
    dyb = DyReLU(64, 2, 4, 21 ** 2 / 3, 10, lambdas=[1.09124] * 2  + [0.5] * 2, init_vals=[1] + [1.09124] * 3, type="b")
    dyc = DyReLU(64, 2, 4, 21 ** 2 / 3, 10, lambdas=[1.09124] * 2  + [0.5] * 2, init_vals=[1] + [1.09124] * 3, type="c")
    print(dya(t).size())
    print(dyb(t).size())
    print(dyc(t).size())
