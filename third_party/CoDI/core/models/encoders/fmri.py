from typing import List
import os

import torch
import torch.nn as nn
import numpy as np
from functools import partial
# from core.models.common.get_model import register
from einops import rearrange
from torch.nn import functional as F

# import sys
# file_path = os.path.abspath(__file__)
# proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
# import pdb; pdb.set_trace()
# sys.path.append(os.path.join(proj_root, 'third_party/mind_vis'))
# sys.path.append(os.path.join(proj_root, 'third_party/mind_vis/code'))
# from third_party.mind_vis.code.sc_mbm.mae_for_fmri import fmri_encoder
# from third_party.mind_vis.code.config import Config_MBM_finetune

version = '0'
symbol = 'fmri'


def freeze(model):
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False


class FmriEmbedder(nn.Module):
    def __init__(self, adaptor_fmri2image_path='checkpoints/fmri_700.pth', force_type_convert=False):
        super(FmriEmbedder, self).__init__()
        self.force_type_convert = force_type_convert
        # num_voxels = 7604
        num_voxels = 15724
        self.adaptor_fmri2image = nn.Sequential(*[nn.Linear(num_voxels, 1024),
                                                  nn.ReLU(),
                                                  nn.Linear(1024, 768),
                                                  nn.ReLU(),
                                                  nn.Linear(768, 768)])

        self.adaptor_fmri2image_path = adaptor_fmri2image_path
        if not os.path.exists(self.adaptor_fmri2image_path): 
            print(self.adaptor_fmri2image_path, 'not exist.')
        else:
            self.init_fmri_weight()
        self.adaptor_fmri2image.eval()
        freeze(self.adaptor_fmri2image)

    def init_fmri_weight(self):
        state_dict = torch.load(self.adaptor_fmri2image_path)
        # adaptor_state_dict = {k.replace('adaptor_fmri2image.', ''): v for k, v in state_dict['model'].items() if k.startswith('adaptor_fmri2image.')}
        adaptor_state_dict = {k.replace('cond_stage_model_fmri.adaptor_fmri2image.', ''): v for k, v in state_dict['module'].items() if k.startswith('cond_stage_model_fmri.adaptor_fmri2image.')}
        self.adaptor_fmri2image.load_state_dict(adaptor_state_dict, strict=True)
        # import pdb; pdb.set_trace()
        print('load fmri adaptor weight from ', self.adaptor_fmri2image_path)

    @torch.no_grad()
    def forward(self, fmri):
        if self.force_type_convert:
            fmri = fmri.half()
        fmri_feat = fmri
        text_feat = self.adaptor_fmri2image(fmri_feat)
        # text_feat = F.normalize(text_feat, dim=-1, p=2)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        # import pdb; pdb.set_trace()
        return text_feat
