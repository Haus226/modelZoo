'''
Title
Making Convolutional Networks Shift-Invariant Again

References
http://arxiv.org/abs/1904.11486
https://github.com/adobe/antialiased-cnns/blob/master/antialiased_cnns/blurpool.py
'''



import torch
from torch import nn
import torch.nn.functional as F

class BlurPool(nn.Module):
    def __init__(self, kernel_size, stride, padding_mode="constant", pad_off=0):
        super(BlurPool, self).__init__()
        self.kernel_size = kernel_size
        '''
        Considering both odd and even kernel_size
        For odd, padding == kernel_size // 2
        For even, left_padding = (kernel_size - 1) // 2, right_padding = kernel_size // 2
        ''' 
        self.padding = [(kernel_size - 1) // 2, kernel_size // 2, (kernel_size - 1) // 2, kernel_size // 2]

        self.padding_mode = padding_mode
        self.padding = [p + pad_off for p in self.padding]
        self.stride = stride

        row = [1]
        for k in range(1, kernel_size):
            value = row[-1] * (kernel_size - k) // k 
            row.append(value)

        self.weight = torch.Tensor(row)[:, None] * torch.Tensor(row)[None, :]
        self.weight = self.weight / torch.sum(self.weight)

    def forward(self, x):
        return F.conv2d(F.pad(x, self.padding, self.padding_mode), 
                        self.weight.repeat(x.size(1), 1, 1, 1), 
                        None, self.stride, groups=x.size(1))

if __name__ == "__main__":
    torch.manual_seed(226)
    # Verifying that strided max pooling is equivalent to densely evaluated 
    # max pooling followed by sub-sampling (stride)
    t = torch.rand((32, 64, 21, 21))
    stride = 2
    print(torch.allclose(nn.MaxPool2d(2, stride=stride)(t), nn.MaxPool2d(2, stride=1)(t)[:, :, ::stride, ::stride]))
    pool = BlurPool(6, 1)
    print(pool(t).size())