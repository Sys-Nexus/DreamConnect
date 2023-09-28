from typing import List
import os

import torch
import torch.nn as nn
import numpy as np
from functools import partial
from einops import rearrange
from torch.nn import functional as F

import sys
file_path = os.path.abspath(__file__)
proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
# import pdb; pdb.set_trace()

version = '0'
symbol = 'fmri'


def freeze(model):
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False


# @register('fmri', version)
class FmriEmbedder(nn.Module):
    def __init__(self, adaptor_fmri2image_path=''):
        super(FmriEmbedder, self).__init__()
        # num_voxels = 7604 # use roi ventral region
        num_voxels = 5917 # use roi ventral region

        self.adaptor_fmri2image = nn.Sequential(*[nn.Linear(num_voxels, 1024),
                                                  nn.ReLU(),
                                                  nn.Linear(1024, 768),
                                                  nn.ReLU(),
                                                  nn.Linear(768, 768)])
        self.adaptor_fmri2image_path = adaptor_fmri2image_path
        if os.path.exists(adaptor_fmri2image_path):
            self.init_fmri_weight()
        # self.adaptor_fmri2image.eval()
        # freeze(self.adaptor_fmri2image)

    def init_fmri_weight(self):
        state_dict = torch.load(self.adaptor_fmri2image_path)
        adaptor_state_dict = {k.replace('adaptor_fmri2image.', ''): v for k, v in state_dict['model'].items() if k.startswith('adaptor_fmri2image.')}
        self.adaptor_fmri2image.load_state_dict(adaptor_state_dict, strict=True)
        # import pdb; pdb.set_trace()
        print('load fmri adaptor weight from ', self.adaptor_fmri2image_path)

    @torch.no_grad()
    def forward(self, fmri_feat):
        text_feat = self.adaptor_fmri2image(fmri_feat)
        # text_feat = F.normalize(text_feat, dim=-1, p=2)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        # import pdb; pdb.set_trace()
        return text_feat

