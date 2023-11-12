import sys
import os
import os.path as osp
import PIL
from PIL import Image
from pathlib import Path
import numpy as np
import numpy.random as npr
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvtrans
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD
from lib.experiments.sd_default import color_adjust, auto_merge_imlist
from torch.utils.data import DataLoader, Dataset

from lib.model_zoo.vd import VD
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml
from lib.model_zoo.openaimodel import UNetModelVD
from lib.model_zoo.diffusion_utils import checkpoint, conv_nd, linear, avg_pool_nd, \
                                          zero_module, normalization, timestep_embedding
from lib.model_zoo.openaimodel import TimestepEmbedSequential
from lib.model_zoo.vd import VDCLIP

import matplotlib.pyplot as plt
from skimage.transform import resize, downscale_local_mean

from ldm.util import log_txt_as_img
from ldm.models.diffusion.fmri_ddpm_edit import LatentDiffusion, DDPM
import torchvision
import k_diffusion as K


class VersatileNetAdaptor(UNetModelVD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dims = self.dims = 2
        model_channels = self.model_channels
        channel_mult = self.channel_mult
        second_dim = self.second_dim

        self.zero_convs = nn.ModuleList([self.make_zero_conv(self.model_channels)])
        for level_idx, (mult, sdim) in list(enumerate(zip(channel_mult, second_dim)))[::-1]:
            for block_idx in range(self.num_noattn_blocks[level_idx] + 1):
                ch = mult * self.model_channels
                self.zero_convs.append(self.make_zero_conv(ch))

        self.middle_block_out = self.make_zero_conv(ch)

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward_dc(self, x, hint, timesteps, c0, c1, xtype, c0_type, c1_type, mixed_ratio):
        hs, outs = [], []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        
        x=x.half()
        emb = self.time_embed(t_emb.half())

        if xtype == 'text':
            x = x[:, :, None, None]
        h = x
        for i_module, t_module in zip(self.unet_image.input_blocks, self.unet_text.input_blocks):
            h = self.mixed_run_dc(i_module, t_module, h, emb, c0, c1, xtype, c0_type, c1_type, mixed_ratio)
            hs.append(h)
        h = self.mixed_run_dc(
            self.unet_image.middle_block, self.unet_text.middle_block, 
            h, emb, c0, c1, xtype, c0_type, c1_type, mixed_ratio)
        for i_module, t_module, zero_conv in zip(self.unet_image.output_blocks, self.unet_text.output_blocks, self.zero_convs):
            h = th.cat([h, hs.pop()], dim=1)
            h = self.mixed_run_dc(i_module, t_module, h, emb, c0, c1, xtype, c0_type, c1_type, mixed_ratio)
            outs.append(zero_conv(h, emb))

        if xtype == 'image':
            return self.unet_image.out(h), outs
        elif xtype == 'text':
            return self.unet_text.out(h).squeeze(-1).squeeze(-1), outs


class VersatileDualStream(LatentDiffusion):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_input(self, batch, k):
        pass

    @torch.no_grad()
    def log_images(self, batch, epoch_n, iter_n, batch_idx, model_wrap, model_wrap_cfg,
                   save_dir, split,
                   cfg_text=7.5, cfg_fmri=1.5,
                   N=2, n_row=4, sample=True, 
                   steps=100, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=False):
        pass

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        pass
