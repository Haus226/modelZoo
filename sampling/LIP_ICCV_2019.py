'''
Title
LIP: Local Importance-based Pooling

References
http://arxiv.org/abs/1908.04156
'''



import torch
from torch import nn
import torch.nn.functional as F

class LIP(nn.AvgPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, 
                ceil_mode=False, count_include_pad=True, divisor_override=None):
        super(LIP, self).__init__(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
    
    def forward(self, x, logits):
        '''
        Logits can be produced by a subnetwork.
        '''
        weight = logits.exp()
        return super(LIP, self).forward(x * weight) / super(LIP, self).forward(weight)
    
if __name__ == "__main__":
    t = torch.arange(64).view(1, 1, 8, 8).float()
    lip = LIP(2)
    print(torch.allclose(lip(t, torch.zeros_like(t)), F.avg_pool2d(t, 2)))
