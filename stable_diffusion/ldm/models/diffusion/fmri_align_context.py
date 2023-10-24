"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

# File modified by authors of InstructPix2Pix from original (https://github.com/CompVis/stable-diffusion).
# See more details in LICENSE.

# Modified by Zigang Geng (zigang@mail.ustc.edu.cn)

import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from functools import partial
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid
import torch.nn.functional as F
from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler
from timm.models.layers import trunc_normal_


__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class NNParams(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(dim), requires_grad=True)
        trunc_normal_(self.cls_token, mean=0., std=10, a=-10, b=10)

    def forward(self):
        return self.cls_token


class FMRIAlign(nn.Module):
    """main class"""
    def __init__(self,
                 cond_stage_config_fmri,
                 cond_stage_config,
                 first_stage_config=None,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 fmri_cond_stage_key="fmri_edit",
                 cond_stage_trainable=False,
                 cond_stage_trainable_fmri=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 cond_stage_forward_fmri=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 deepspeed="",
                 is_fmri_input=True,
                 scheduler_config=None,
                 *args, **kwargs):
        super().__init__()

        self.deepspeed = deepspeed
        self.frmi_cond_stage_key = fmri_cond_stage_key
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        # import pdb; pdb.set_trace()
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_trainable_fmri = cond_stage_trainable_fmri
        
        # import pdb; pdb.set_trace()
        self.instantiate_cond_stage(cond_stage_config)
        self.instantiate_cond_stage_fmri(cond_stage_config_fmri)
        
        self.cond_stage_key = cond_stage_key
        self.is_fmri_input = is_fmri_input
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        
        self.cond_stage_forward = cond_stage_forward
        self.cond_stage_forward_fmri = cond_stage_forward_fmri

        self.clip_denoised = False
        self.bbox_tokenizer = None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

        self.additional_loss_type = kwargs.pop("additional_loss_type", None)

    def instantiate_cond_stage_fmri(self, config):
        # import pdb; pdb.set_trace();
        if not self.cond_stage_trainable_fmri:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model_fmri = self.first_stage_model_fmri
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model_fmri = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model_fmri = model.eval()
                self.cond_stage_model_fmri.train = disabled_train
                for param in self.cond_stage_model_fmri.parameters():
                    param.requires_grad = False
            print('fMRI Cond Stage is Freezed.')
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model_fmri = model
            print('fMRI Cond Stage is Trainable.')

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                # import pdb; pdb.set_trace()
                ckpt_path = config['params']['ckpt_path']
                if os.path.exists(ckpt_path):
                    codi_clip_ckpt = torch.load(ckpt_path, map_location='cpu')
                    codi_clip_ckpt = {k.replace('clip.',''):codi_clip_ckpt[k] for k in codi_clip_ckpt.keys() if k.startswith('clip.model.')}
                    model.load_state_dict(codi_clip_ckpt, strict=True)
                    # model.encode_type = 'encode_vision'
                    model.encode_type = 'encode_text'

                self.cond_stage_model = model
                self.cond_stage_model.eval()
                # self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model


    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        if os.path.exists(path):
            sd = torch.load(path, map_location="cpu")
            if "state_dict" in list(sd.keys()):
                sd = sd["state_dict"]
            keys = list(sd.keys())

            self_sd = self.state_dict()

            for k in keys:
                for ik in ignore_keys:
                    if k.startswith(ik):
                        print("Deleting key {} from state_dict.".format(k))
                        del sd[k]
            missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
                sd, strict=False)
            print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
            # if len(missing) > 0:
            #     print(f"Missing Keys: {missing}")
            # if len(unexpected) > 0:
            #     print(f"Unexpected Keys: {unexpected}")
            # import pdb; pdb.set_trace()
        else:
            warnings.warn("The pre-trained stable diffusion model has not been loaded. "
                "If you are in the training phase, please check your code. "
                "If you are in the testing phase, you can ignore this warning.")


    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def get_learned_conditioning_fmri(self, cc):
        if self.cond_stage_forward_fmri is None:
            if hasattr(self.cond_stage_model_fmri, 'encode') and callable(self.cond_stage_model_fmri.encode):
                c = self.cond_stage_model_fmri.encode(cc)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                # import pdb; pdb.set_trace();
                c = self.cond_stage_model_fmri(cc)
        else:
            assert hasattr(self.cond_stage_model_fmri, self.cond_stage_forward_fmri)
            c = getattr(self.cond_stage_model_fmri, self.cond_stage_forward_fmri)(cc)
        return c

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c, is_return_pool=is_return_pool)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def forward(self, batch, batch_idx, num_steps, *args, **kwargs):
        # x, c = self.get_input(batch, self.first_stage_key)
        fmri = batch['fmri'].cuda()
        caps = batch['cap']
        # import pdb; pdb.set_trace();

        if len(fmri.shape)>2:
            repeat_index = batch_idx % 3
            fmri = fmri[:, repeat_index]
        # gt_image = batch['image'].cuda()
        fmri_embed = self.get_learned_conditioning_fmri(fmri)
        with torch.no_grad():
            caps_embed = self.get_learned_conditioning(caps)
        caps_embed = caps_embed.detach().requires_grad_(True)
        # caps_pool_output = caps_pool_output.detach().requires_grad_(True)
        import pdb; pdb.set_trace();
        # text_embed = torch.mean(caps_embed, dim=1) @ self.cond_stage_model_fmri.text_projection
        loss = torch.nn.L1Loss()(fmri_embed.reshape(*caps_embed.shape), caps_embed)
        if False: # maybe in future include some contrastive loss
            text_embed = caps_pool_output @ self.cond_stage_model_fmri.text_projection
            fmri_embed = fmri_embed.squeeze(1) @ self.cond_stage_model_fmri.image_projection

            # normalized features
            fmri_embed = F.normalize(fmri_embed, dim=-1, p=2)
            text_embed = F.normalize(text_embed, dim=-1, p=2)

            # cosine similarity as logits
            logit_scale = self.cond_stage_model_fmri.logit_scale.exp()
            logits_per_fmri = logit_scale * fmri_embed @ text_embed.t()
            logits_per_text = logit_scale * text_embed @ fmri_embed.t()

            local_batch_size = fmri_embed.shape[0]
            self.labels = torch.arange(local_batch_size, device=fmri_embed.device)

            loss = (F.cross_entropy(logits_per_fmri, self.labels) + \
                F.cross_entropy(logits_per_text, self.labels)) / 2

        # loss = torch.nn.L1Loss()(fmri_embed, pool_caps_embed)
        loss_dict = {'L1': loss.item()}

        return loss, loss_dict
