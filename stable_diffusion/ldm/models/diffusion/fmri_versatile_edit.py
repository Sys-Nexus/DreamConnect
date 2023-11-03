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
        pth = 'third_party/versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
        cfgm = model_cfg_bank()(cfgm_name)
        net = get_model()(cfgm)
        sd = torch.load(pth, map_location='cpu')
        net.load_state_dict(sd, strict=False)    

        # Might require editing the GPU assignments due to Memory issues
        net.clip.cuda(0)
        net.autokl.cuda(0)

        #net.model.cuda(1)
        sampler = sampler(net)

        n_samples = 1
        ddim_steps = 50
        ddim_eta = 0
        scale = 7.5
        strength = 0.75
        mixing = 0.4
        t_enc = int(strength * ddim_steps)

        h, w = 512,512
        shape = [n_samples, 4, h//8, w//8]

        xtype = 'image'
        ctype = 'prompt'
        net.autokl.half()

        # sampler.model.model.diffusion_model.device='cuda:1'
        sampler.model.model.diffusion_model.half()
        
        self.sampler = sampler
        self.net = net
        self.net.eval()

        self.device = 'cuda:0'
        self.t_enc = t_enc
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.scale = scale
        self.mixing = mixing

    @torch.no_grad()    
    def log_images(self, batch, epoch_n, iter_n, batch_idx, model_wrap, model_wrap_cfg,
                   save_dir, split,
                   cfg_text=7.5, cfg_fmri=1.5,
                   N=2, n_row=4, sample=True, 
                   steps=100, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=False):
        # self.model.eval()
        
        # zim = Image.open('results/vdvae/subj{:02d}/{}.png'.format(sub,im_id))
   
        # zim = regularize_image(zim)
        # import pdb; pdb.set_trace();
        zim = batch['image']
        # zin = zim*2 - 1
        zin = zim.half().cuda()

        init_latent = self.net.autokl_encode(zin)
        
        self.sampler.make_schedule(ddim_num_steps=self.ddim_steps, ddim_eta=self.ddim_eta, verbose=False)

        dummy = ''
        utx = self.net.clip_encode_text(dummy)
        utx = utx.half()
        
        dummy = torch.zeros((1,3,224,224))
        uim = self.net.clip_encode_vision(dummy)
        uim = uim.half()
        
        cim = self.net.clip_encode_vision(zim)
        cap = batch['cap'][0]
        ctx = self.net.clip_encode_text(cap)

        z_enc = self.sampler.stochastic_encode(init_latent, torch.tensor([self.t_enc]).cuda())
        z = self.sampler.decode_dc(
            x_latent=z_enc,
            first_conditioning=[uim, cim],
            second_conditioning=[utx, ctx],
            t_start=self.t_enc,
            unconditional_guidance_scale=self.scale,
            xtype='image', 
            first_ctype='vision',
            second_ctype='prompt',
            mixed_ratio=(1-self.mixing), )
        
        x = self.net.autokl_decode(z)
        x = torch.clamp((x+1.0)/2.0, min=0.0, max=1.0)
        import pdb; pdb.set_trace();