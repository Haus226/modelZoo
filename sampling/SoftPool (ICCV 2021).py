'''
Title
Refining activation downsampling with SoftPool

References
http://arxiv.org/abs/2101.00440
'''



import torch
from torch import nn

class SoftPool(nn.AvgPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, 
                ceil_mode=False, count_include_pad=True, divisor_override=None):
        super(SoftPool, self).__init__(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
    
    def forward(self, x):
        '''
        Logits can be produced by a subnetwork.
        '''
        weight = x.exp()
        return super(SoftPool, self).forward(x * weight) / super(SoftPool, self).forward(weight)
    
if __name__ == "__main__":
    t = torch.arange(64).view(1, 1, 8, 8).float()
    softpool = SoftPool(2)
    print(softpool(t))
