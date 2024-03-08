import os
import sys
import torch
import cv2
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm
from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F
from enum import Enum
import numpy as np
import torch.nn.init as init



class Deformator_align(nn.Module):
    def __init__(self, input_dim=None, out_dim=None, inner_dim=1024,):
        super(Deformator_align, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(self.input_dim, inner_dim)
        self.bn1 = nn.BatchNorm1d(inner_dim)
        self.act1 = nn.ELU()
        self.fc2 = nn.Linear(inner_dim, inner_dim)
        self.bn2 = nn.BatchNorm1d(inner_dim)
        self.act2 = nn.ELU()
        self.fc3 = nn.Linear(inner_dim, inner_dim)
        self.bn3 = nn.BatchNorm1d(inner_dim)
        self.act3 = nn.ELU()
        self.fc4 = nn.Linear(inner_dim, self.out_dim)

    def forward(self, input):
        x1 = self.fc1(input)
        x = self.act1(self.bn1(x1))
        x2 = self.fc2(x)
        x = self.act2(self.bn2(x2 + x1))
        x3 = self.fc3(x)
        x = self.act3(self.bn3(x3 + x2 + x1))
        out = self.fc4(x)
        return out


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size
    def forward(self, tensor):
        return tensor.view(self.size)
def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class VAE_align(nn.Module):
    def __init__(self, input_dim=None, out_dim=None, z_dim=10, nc=3):
        super(VAE_align, self).__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.encoder = nn.Sequential(
            nn.Conv2d( self.input_dim, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),                 # B, 256
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim*2, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, self.out_dim, 4, 2, 1),  # B, nc, 64, 64
        )
        self.weight_init()
    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)
    def forward(self, x):
        distributions = self._encode(x)
        x_recon = self._decode(distributions)
        return x_recon 
    def _encode(self, x):
        return self.encoder(x)
    def _decode(self, z):
        return self.decoder(z)


class align_block(nn.Module):
    def __init__(self, input_dim=None, out_dim1=None, out_dim2=None, inner_dim=512, out_channel=768):
        super(align_block, self).__init__()
        self.inner_dim = inner_dim
        self.input_dim = input_dim
        self.out_dim1 = out_dim1*out_channel
        self.out_dim2 = out_dim2*out_channel
        self.out_channel = out_channel
        self.out_spatial1 = out_dim1
        self.out_spatial2 = out_dim2
        
        self.encoder1 = nn.Linear(1024, self.inner_dim)  ##  15724  192
        self.encoder3 = nn.Linear( self.inner_dim, self.inner_dim)
        self.encoder5 = nn.Linear( self.inner_dim, self.inner_dim)
        
        self.x_branch1 =  nn.Linear(self.inner_dim, self.out_dim1)
        self.x_branch8 = nn.ELU()
        self.cap_branch1 = nn.Linear(self.inner_dim, self.out_dim2)
        self.cap_branch8 = nn.ELU()

        
    def forward(self, x):
        
        if len(x.shape)>2:
            x = torch.mean(x, dim=1)
        else:
            x = x
        B, feature = x.shape

        x = F.interpolate(x.unsqueeze(1), size=(1024), mode='linear').squeeze(1)
        distributions = self.encoder1(x)
        distributions = self.encoder3(distributions)
        distributions = self.encoder5(distributions)

        predict_x = self.x_branch1(distributions)
        predict_x = self.x_branch8(predict_x).view(B, self.out_spatial1, self.out_channel)#.permute(0,2,1)

        predict_cap  = self.cap_branch1(distributions)
        predict_cap  = self.cap_branch8(predict_cap).view(B, self.out_spatial2, self.out_channel)#.permute(0,2,1)

        return  predict_x, predict_cap
