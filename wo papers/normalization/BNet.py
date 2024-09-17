'''
Title
Batch Normalization with Enhanced Linear Transformation

References
http://arxiv.org/abs/2011.14150
'''



from torch import nn
class BNet(nn.Module):
    def __init__(self, channels, kernel_size):
        '''
        In standard BN, each channel is scaled and shifted using channel-specific parameters, 
        where the output is computed as y_c = gamma_c * x_c + beta_c
        Instead of applying the same scaling factor across all elements in a channel, 
        the paper proposes to utilize a convolutional operation with bias for the scaling and shifting.
        This approach allows the scaling to vary based on local spatial information within each channel, 
        while maintaining the property of BN that treats each channel independently 
        but applies the same kernel across spatial dimensions.
        '''

        self.bn = nn.BatchNorm2d(channels, affine=False)
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2,
                            groups=channels)
    
    def forward(self, x):
        return self.conv(self.bn(x))
