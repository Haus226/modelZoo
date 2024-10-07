import torch,math 
import torch.nn.functional as F
from torch import nn
import torch

def get_1d_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i + 0.5) / L) / math.sqrt(L)
    if freq == 0: 
        return result 
    else: 
        return result * math.sqrt(2) 

def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                    'bot1','bot2','bot4','bot8','bot16','bot32',
                    'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    
    if 'top' in method:
        # Based on experiments
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        # Start from top left corner of 2D DCT
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        # Based on experiments
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

def get_dct_weights(height, width, input_channels, 
                    u_idx, v_idx):
    # Top-16 frequency components
    # u_idx : horizontal indices of selected fequency 
    # v_idx : vertical indices of selected fequency 
    assert input_channels % len(u_idx) == 0
    scale_ratio = width // 7
    u_idx = [u * scale_ratio for u in u_idx]
    v_idx = [v * scale_ratio for v in v_idx]
    # Make the frequencies in different sizes are identical to a 7x7 frequency space used in the paper
    # eg, (2, 2) in 14x14 is identical to (1, 1) in 7x7

    dct_weights = torch.zeros(input_channels, height, width) 
    c_part = input_channels // len(u_idx) # Split channels for multi spectral attention
    for i, (u, v) in enumerate(zip(u_idx, v_idx)): 
        for h in range(height):
            for w in range(width): 
                dct_weights[i * c_part: (i + 1) * c_part, h, w] = get_1d_dct(h, u, width) * get_1d_dct(w, v, height) 
    return dct_weights 

class FCA(nn.Module):
    def __init__(self,
                input_channels,
                r, height, width, method="top16"):
        super(FCA, self).__init__()
        
        self.width = width
        self.height = height
        u_idx, v_idx = get_freq_indices(method)
        self.weights = get_dct_weights(self.width, self.height, input_channels, u_idx, v_idx)
        self.fc = nn.Sequential(
            nn.Linear(input_channels, input_channels // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_channels // r, input_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # This is for compatibility in instance segmentation and object detection.
        # y = F.adaptive_avg_pool2d(x, (self.height, self.width))
        att = torch.sum(x * self.weights, dim=(2, 3))
        att = self.fc(att).view(b, c, 1, 1)
        return x * att

if __name__ == "__main__":
    t = torch.arange(27).view(1, 3, 3, 3).float()
    fca = FCA(3, 4, 3, 3, "top1")
    print(fca.forward(t).size())