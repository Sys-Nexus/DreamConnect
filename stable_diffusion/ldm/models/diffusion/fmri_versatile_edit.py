import sys
import os
import os.path as osp
import PIL
from PIL import Image
from pathlib import Path
import numpy as np
import numpy.random as npr

import torch
import torchvision.transforms as tvtrans
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD
from lib.experiments.sd_default import color_adjust, auto_merge_imlist
from torch.utils.data import DataLoader, Dataset

from lib.model_zoo.vd import VD
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml
import matplotlib.pyplot as plt
from skimage.transform import resize, downscale_local_mean

from ldm.models.diffusion.fmri_ddpm_edit import LatentDiffusion


def regularize_image(x):
        BICUBIC = PIL.Image.Resampling.BICUBIC
        if isinstance(x, str):
            x = Image.open(x).resize([512, 512], resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, PIL.Image.Image):
            x = x.resize([512, 512], resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, np.ndarray):
            x = PIL.Image.fromarray(x).resize([512, 512], resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, torch.Tensor):
            pass
        else:
            assert False, 'Unknown image type'

        assert (x.shape[1]==512) & (x.shape[2]==512), \
            'Wrong image size'
        return x


class fMRIVersatileEdit(LatentDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfgm_name = 'vd_noema'
        sampler = DDIMSampler_VD
        pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
        cfgm = model_cfg_bank()(cfgm_name)
        net = get_model()(cfgm)
        sd = torch.load(pth, map_location='cpu')
        net.load_state_dict(sd, strict=False)    

        # Might require editing the GPU assignments due to Memory issues
        net.clip.cuda(0)
        net.autokl.cuda(0)

        #net.model.cuda(1)
        sampler = sampler(net)
    
    # def 