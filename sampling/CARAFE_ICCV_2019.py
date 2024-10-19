'''
Title
CARAFE: Content-Aware ReAssembly of FEatures

References
http://arxiv.org/abs/1905.02188
https://github.com/XiaLiPKU/CARAFE/blob/master/carafe.py
'''



import torch
from torch import nn
from torch.nn import functional as F

class CARAFE(nn.Module):
    def __init__(self, in_channels, mid_channels, scale, kernel_encoder, kernel_up, up_mode="nearest"):
        super(CARAFE, self).__init__()
        self.scale = scale
        self.comp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        self.enc = nn.Sequential(
            nn.Conv2d(mid_channels, (scale * kernel_up) ** 2, kernel_encoder, padding=kernel_encoder // 2),
            nn.BatchNorm2d((scale * kernel_up) ** 2),
        )
        self.pix_shf = nn.PixelShuffle(scale)
        self.upsmp = nn.Upsample(scale_factor=scale, mode=up_mode)
        # Refer to formula in nn.Conv2d where H_out = H_in, stride=1 to compute the padding needed.
        self.unfold = nn.Unfold(kernel_size=kernel_up, dilation=scale, 
                                padding=kernel_up // 2 * scale)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale
        
        W = self.comp(X)                                
        W = self.enc(W)                                 
        W = self.pix_shf(W)                             
        W = F.softmax(W, dim=1)   

        X = self.upsmp(X)      
        X = self.unfold(X)             

        X = X.view(b, c, -1, h_, w_)                    
        X = torch.einsum('bkhw, bckhw->bchw', [W, X])    
        return X


if __name__ == '__main__':
    torch.manual_seed(226)
    t = torch.Tensor(32, 64, 21, 21)
    carafe = CARAFE(64, 32, 2, 3, 5)
    print(carafe(t).size())