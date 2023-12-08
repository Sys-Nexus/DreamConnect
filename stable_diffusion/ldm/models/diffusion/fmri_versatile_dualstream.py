import sys
import os
import os.path as osp
import PIL
from PIL import Image
from pathlib import Path
import numpy as np
import numpy.random as npr
from einops import rearrange, repeat
import copy

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvtrans
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD
from lib.model_zoo.ddim_vd_dual import DDIMSampler_Dual
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

from ldm.util import log_txt_as_img, instantiate_from_config
from ldm.models.diffusion.fmri_ddpm_edit import LatentDiffusion, DDPM
import torchvision
import k_diffusion as K

from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.modules.attention import SpatialTransformer

class FusionPriorUnetModel(UNetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        channel_mult = self.channel_mult
        num_res_blocks = self.num_res_blocks
        attention_resolutions = self.attention_resolutions
        context_dim = self.context_dim
        transformer_depth = self.transformer_depth
        default_eps = self.default_eps
        force_type_convert = self.force_type_convert
        num_head_channels = self.num_head_channels
        model_channels = self.model_channels
        num_heads = self.num_heads
        legacy = self.legacy
        
        # print(self.transformer_depth)
        # import pdb; pdb.set_trace()
        transformer_depth = 1

        is_simple_basic = True
        SpatialTransformerCLS = SpatialTransformer

        self.merge_blocks = nn.ModuleList([])
        merge_blocks_list = []
        input_block_chans = [model_channels]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = []
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    # print('heads: ', num_heads, dim_head)
                    layers.append(
                        SpatialTransformerCLS(
                            ch, num_heads, dim_head, default_eps=default_eps, force_type_convert=force_type_convert, depth=transformer_depth, context_dim=ch, use_linear=True, is_simple_basic=is_simple_basic
                        )
                    )
                merge_blocks_list.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                # print('heads: ', num_heads, dim_head)
                layers = [SpatialTransformerCLS(ch, num_heads, dim_head, default_eps=default_eps, force_type_convert=force_type_convert, 
                                            depth=transformer_depth, context_dim=ch, use_linear=True, is_simple_basic=is_simple_basic)]
                merge_blocks_list.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
                ch = out_ch
                ds *= 2
        # import pdb; pdb.set_trace()
        # print('heads: ', num_heads, dim_head)
        layers = [SpatialTransformerCLS(ch, num_heads, dim_head, default_eps=default_eps, 
                                                force_type_convert=force_type_convert, 
                                                depth=transformer_depth,
                                                context_dim=ch,
                                                use_linear=True,
                                                is_simple_basic=is_simple_basic)]
        
        merge_blocks_list.append(TimestepEmbedSequential(*layers))
        merge_blocks_list = list(reversed(merge_blocks_list))
        for block in merge_blocks_list: self.merge_blocks.append(block)

        # for level, mult in list(enumerate(channel_mult))[::-1]:
        #     for i in range(num_res_blocks + 1):
        #         ch = model_channels * mult
        #         ich = input_block_chans.pop()
        #         if ds in attention_resolutions:
        #             if num_head_channels == -1:
        #                 dim_head = ch // num_heads
        #             else:
        #                 num_heads = ch // num_head_channels
        #                 dim_head = num_head_channels
        #             if legacy:
        #                 #num_heads = 1
        #                 dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        #             layers = [SpatialTransformer(ch, num_heads, dim_head, default_eps=default_eps, force_type_convert=force_type_convert, 
        #                                         depth=transformer_depth, context_dim=ch)]
        #             self.merge_blocks.append(TimestepEmbedSequential(*layers))

        #     if level and i == num_res_blocks:
        #         out_ch = ch
        #         layers = [SpatialTransformer(ch, num_heads, dim_head, default_eps=default_eps, force_type_convert=force_type_convert, 
        #                                     depth=transformer_depth, context_dim=ch)]
        #         self.merge_blocks.append(TimestepEmbedSequential(*layers))
        #         ds //= 2

        print('len of merge blocks is {}.'.format(len(self.merge_blocks)))
        # import pdb; pdb.set_trace()

    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, num_control_layers=10, **kwargs):
        if control is not None:
            hs = []
            with torch.no_grad():
                t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
                emb = self.time_embed(t_emb.type(self.time_embed[0].weight.dtype))
                h = x.type(self.dtype)
                for i, module in enumerate(self.input_blocks):
                    h = module(h, emb, context)
                    hs.append(h)
                h = self.middle_block(h, emb, context)
            
            if control is not None:
                # h += control.pop(0)
                cat = self.merge_blocks[0](torch.cat([h,control.pop(0)],dim=-1), emb)
                h = cat[:,:,:,:h.shape[-1]] # + torch.zeros_like(cat[:,:,:,:h.shape[-1]])*cat[:,:,:,h.shape[-1]:]
                # import pdb; pdb.set_trace()

            for i, module in enumerate(self.output_blocks):
                if only_mid_control or control is None or i > num_control_layers:# or i in unmatched_layers:
                    h = torch.cat([h, hs.pop()], dim=1)
                else:
                    # mix = torch.cat([h, hs.pop() + control.pop(0)], dim=1)
                    # mix = hs.pop() + control.pop(0)
                    # print(i, hs[-1].shape, control[0].shape, h.shape)
                    cat = self.merge_blocks[i+1](torch.cat([hs.pop(),control.pop(0)],dim=-1), emb)
                    mix = cat[:,:,:,:h.shape[-1]] # + torch.zeros_like(cat[:,:,:,:h.shape[-1]])*cat[:,:,:,h.shape[-1]:]
                    # import pdb; pdb.set_trace()
                    h = torch.cat([h, mix], dim=1)
                h = module(h, emb, context)

            # import pdb; pdb.set_trace()

            h = h.type(x.dtype)
            return self.out(h)
        else:
            # import pdb; pdb.set_trace();
            return super().forward(x, timesteps=timesteps, context=context, **kwargs)


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, num_control_layers=8, **kwargs):
        # print(x.shape, timesteps)
        if control is not None:
            # unmatched_layers = [2, 5, 8]
            hs = []
            with torch.no_grad():
                t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
                emb = self.time_embed(t_emb.type(self.time_embed[0].weight.dtype))
                h = x.type(self.dtype)
                for i, module in enumerate(self.input_blocks):
                    # print('gen: ', i, h.shape, 'context: ', context.shape)
                    h = module(h, emb, context)
                    hs.append(h)
                # print('gen: ', i+1, h.shape, 'context: ', context.shape)
                h = self.middle_block(h, emb, context)
                # print('gen: ', i+2, h.shape, 'context: ', context.shape)

            # import pdb; pdb.set_trace()

            if control is not None:
                h += control.pop(0)

            for i, module in enumerate(self.output_blocks):
                # print(i, hs[-1].shape, control[0].shape)
                # if i in unmatched_layers:
                #     print('out i {}:'.format(i))
                #     print(control[0].shape)
                #     control.pop(0)

                if only_mid_control or control is None or i > num_control_layers:# or i in unmatched_layers:
                    # print(h.shape, hs[-1].shape)
                    h = torch.cat([h, hs.pop()], dim=1)
                else:
                    # print('insert: ', i, hs[-1].shape, control[0].shape)
                    h = torch.cat([h, hs.pop() + control.pop(0)], dim=1)
                h = module(h, emb, context)

            h = h.type(x.dtype)
            return self.out(h)
        else:
            # import pdb; pdb.set_trace();
            return super().forward(x, timesteps=timesteps, context=context, **kwargs)

class PostVersatileNetAdaptor(UNetModelVD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dims = self.dims = 2
        model_channels = self.model_channels
        channel_mult = self.channel_mult

        self.zero_convs = nn.ModuleList([])

        ch = channel_mult[-1] * model_channels
        self.middle_block_out = self.make_zero_conv(ch)

        unmatched_layers = [2, 5, 8]
        out_cs = {2: None, 5: 640, 8: 320}
        stride_i = 0
        for level_idx, mult in list(enumerate(channel_mult))[::-1]:
            for block_idx in range(self.num_noattn_blocks[level_idx] + 1):
                ch = mult * model_channels
                # print('ch: ', ch)
                if stride_i in unmatched_layers:
                    self.zero_convs.append(self.make_zero_conv(ch, kernel_size=2, stride=2, out_channels=out_cs[stride_i]))
                else:
                    self.zero_convs.append(self.make_zero_conv(ch))
                stride_i += 1

    def make_zero_conv(self, channels, kernel_size=1, stride=1, out_channels=None):
        if out_channels is None:
            return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, kernel_size, stride=stride, padding=0)))
        else:
            return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, out_channels, kernel_size, stride=stride, padding=0)))

    def forward_dc(self, x, timesteps, c0, c1, xtype, c0_type, c1_type, mixed_ratio):
        # print(x.shape, c0.shape, c1.shape, timesteps)
        # import pdb; pdb.set_trace()

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
        outs.append(self.middle_block_out(h, emb))
        
        for i, (i_module, t_module, zero_conv) in enumerate(zip(self.unet_image.output_blocks, 
                                                self.unet_text.output_blocks, self.zero_convs)):
            h = th.cat([h, hs.pop()], dim=1)
            h = self.mixed_run_dc(i_module, t_module, h, emb, c0, c1, xtype, c0_type, c1_type, mixed_ratio)
            out_i = zero_conv(h, emb)
            outs.append(out_i)
            # print(i, h.shape, out_i.shape)

        if xtype == 'image':
            return self.unet_image.out(h), outs
        elif xtype == 'text':
            return self.unet_text.out(h).squeeze(-1).squeeze(-1), outs

class PreVersatileNetAdaptor(UNetModelVD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dims = self.dims = 2
        model_channels = self.model_channels
        channel_mult = self.channel_mult

        # self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)]) # different from postversatilenetadaptor

        # ch = channel_mult[-1] * model_channels
        # self.middle_block_out = self.make_zero_conv(ch)

        # stride_i = 0
        # for level, mult in enumerate(channel_mult):
        #     for nr in range(self.num_noattn_blocks[level]):
        #         ch = mult * model_channels
        #         self.zero_convs.append(self.make_zero_conv(ch))
        #     if level != len(channel_mult) - 1:
        #         self.zero_convs.append(self.make_zero_conv(ch))

        # import pdb; pdb.set_trace()

    def make_zero_conv(self, channels, kernel_size=1, stride=1, out_channels=None):
        if out_channels is None:
            return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, kernel_size, stride=stride, padding=0)))
        else:
            return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, out_channels, kernel_size, stride=stride, padding=0)))

    def forward_dc(self, x, timesteps, c0, c1, xtype, c0_type, c1_type, mixed_ratio):
        # print(x.shape, c0.shape, c1.shape, timesteps)
        # import pdb; pdb.set_trace()

        hs, outs = [], []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        
        x=x.half()
        emb = self.time_embed(t_emb.half())

        if xtype == 'text':
            x = x[:, :, None, None]
        h = x
        for i, (i_module, t_module) in enumerate(zip(self.unet_image.input_blocks, self.unet_text.input_blocks)):
            h = self.mixed_run_dc(i_module, t_module, h, emb, c0, c1, xtype, c0_type, c1_type, mixed_ratio)
            outs.append(h)
            hs.append(h)

        # for i, (i_module, t_module, zero_conv) in enumerate(zip(self.unet_image.input_blocks, self.unet_text.input_blocks, self.zero_convs)):
        #     h = self.mixed_run_dc(i_module, t_module, h, emb, c0, c1, xtype, c0_type, c1_type, mixed_ratio)
        #     out_i = zero_conv(h, emb)
        #     outs.append(out_i)
        #     hs.append(h)

        h = self.mixed_run_dc(
            self.unet_image.middle_block, self.unet_text.middle_block, 
            h, emb, c0, c1, xtype, c0_type, c1_type, mixed_ratio)
        # outs.append(self.middle_block_out(h, emb))
        outs.append(h)        
        for i_module, t_module in zip(self.unet_image.output_blocks, self.unet_text.output_blocks):
            h = th.cat([h, hs.pop()], dim=1)
            h = self.mixed_run_dc(i_module, t_module, h, emb, c0, c1, xtype, c0_type, c1_type, mixed_ratio)

        outs = list(reversed(outs))
        # import pdb; pdb.set_trace()
        if xtype == 'image':
            return self.unet_image.out(h), outs
        elif xtype == 'text':
            return self.unet_text.out(h).squeeze(-1).squeeze(-1), outs


class DualLDM(LatentDiffusion):
    def __init__(self, clip_cfg, fmri_vclip_cfg=None, fmri_vclip_pretrain_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vd_clip = VDCLIP(clip_cfg)

        pretrained_control_unet_path = kwargs['pretrained_control_unet_path']
        if pretrained_control_unet_path is not None and os.path.exists(pretrained_control_unet_path):
            # import pdb; pdb.set_trace()
            pretrained_state_dict = torch.load(pretrained_control_unet_path, map_location="cpu")
            pretrained_state_dict = {k:v for k,v in pretrained_state_dict.items() if 'clip.' in k}
            missing, unexpected = self.vd_clip.load_state_dict(pretrained_state_dict, strict=False)

            print('clip missing {} params.'.format(len(missing)))
            print('clip missing: ', missing)
            print('clip unexpected {} params.'.format(len(unexpected)))

        self.sampler = DDIMSampler_Dual(self)

        ddim_steps = 50
        # ddim_steps = 500
        ddim_eta = 0
        scale = 7.5
        strength = 0.75
        mixing = 0.4
        t_enc = int(strength * ddim_steps)

        self.t_enc = t_enc
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.scale = scale
        self.mixing = mixing

        # import pdb; pdb.set_trace();
        if fmri_vclip_cfg is not None:
            self.fmri_vclip_pretrain_path = fmri_vclip_pretrain_path
            self.instantiate_fmri_vclip(fmri_vclip_cfg)

    def instantiate_fmri_vclip(self, config):
        model = instantiate_from_config(config)
        self.fmri_vclip = model.eval()

        for param in self.fmri_vclip.parameters():
            param.requires_grad = False

        # import pdb; pdb.set_trace();
        if self.fmri_vclip_pretrain_path is not None and os.path.exists(self.fmri_vclip_pretrain_path):
            sd = torch.load(self.fmri_vclip_pretrain_path, map_location="cpu")
            state_dict = sd['model_state_dict']
            filter_state_dict = {k.replace('voxel2clip.',''):v for k,v in state_dict.items() if 'voxel2clip.' in k}
            missing, unexpected = self.fmri_vclip.load_state_dict(filter_state_dict, strict=False)

            print('vox2clip vclip: [missing]', len(missing))
            print('vox2clip vclip: [unexpected] ', len(unexpected))
            
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None, uncond=0.075, sz=256):
        x = DDPM.get_input(self, batch, k)
        if bs is not None:
            x = x[:bs]
        if sz is not None:
            x = F.interpolate(x, (sz,sz))
        
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        with torch.no_grad():
            voxel = batch['fmri'].to(z)
            if bs is not None: voxel = voxel[:bs]
            if voxel.shape[1] == 3: voxel = voxel.mean(dim=1)
            self.fmri2lowlevel = self.fmri2lowlevel.float()
            lowlevel_vae = self.fmri2lowlevel(voxel).half()

        cond_key = cond_key or self.cond_stage_key

        xc = DDPM.get_input(self, batch, cond_key)
        cap = batch['cap']
        fmri_vae = lowlevel_vae
        if bs is not None:
            xc["c_crossattn"] = xc["c_crossattn"][:bs]
            xc["c_crossattn_1"] = xc["c_crossattn_1"][:bs]
            xc["c_concat"] = xc["c_concat"][:bs]
            cap = batch['cap'][:bs]
            fmri_vae = lowlevel_vae[:bs]

        if sz is not None:
            xc["c_concat"] = F.interpolate(xc["c_concat"], (sz,sz))
        x, xc["c_concat"], xc["c_crossattn_1"] = x.to(z), xc["c_concat"].to(z), xc["c_crossattn_1"].to(z)

        cond = {}
        random = torch.rand(x.size(0), device=z.device)
        prompt_mask = rearrange(random < uncond, "n -> n 1 1")
        fmri_prompt_mask = (random >= uncond).float() * (random < uncond*2).float()
        fmri_prompt_mask = rearrange(fmri_prompt_mask, "n -> n 1 1")
        input_mask = 1 - rearrange((random >= uncond*2).float() * (random < uncond*3).float(), "n -> n 1 1 1")
        
        null_prompt = self.get_learned_conditioning([""])

        null_x = torch.zeros_like(xc["c_concat"])
        null_cap = ['' for _ in range(len(cap))]
        
        # import pdb; pdb.set_trace();
        # # _, fmri_null_x = self.fmri_vclip(torch.zeros_like(voxel).half())
        # fmri_null_x = self.vd_clip.clip_encode_vision(null_x)
        # fmri_null_cap = self.vd_clip.clip_encode_text(null_cap)

        # # _, fmri_x = self.fmri_vclip(voxel.half())
        # fmri_x = self.vd_clip.clip_encode_vision(x)
        # fmri_cap = self.vd_clip.clip_encode_text(cap)
        # import pdb; pdb.set_trace();
        fmri_null_x = self.vd_clip.clip_encode_vision(null_x)
        fmri_null_cap = self.vd_clip.clip_encode_text(null_cap)
        # fmri_x = batch['nsd_clipvision'][:null_x.shape[0]].to(fmri_null_x)
        # fmri_cap = batch['nsd_cliptext'][:null_x.shape[0]].to(fmri_null_x)
        fmri_x = self.vd_clip.clip_encode_vision(xc["c_concat"])
        fmri_cap = self.vd_clip.clip_encode_text(cap)


        cond["c_crossattn_1"] = {}
        if force_c_encode is False:
            cond["c_crossattn_1"]["image_emb"] = [torch.where(fmri_prompt_mask.bool(), fmri_null_x, fmri_x)]
            cond["c_crossattn_1"]["text_emb"] = [torch.where(fmri_prompt_mask.bool(), fmri_null_cap, fmri_cap)]
            cond["c_crossattn"] = [torch.where(prompt_mask, null_prompt, self.get_learned_conditioning(xc["c_crossattn"]).detach())]
        else:
            cond["c_crossattn_1"]["image_emb"] = [fmri_x]
            cond["c_crossattn_1"]["text_emb"] = [fmri_cap]
            cond["c_crossattn"] = [self.get_learned_conditioning(xc["c_crossattn"]).detach()]

        cond["c_crossattn_1"]["fmri_vae"] = [fmri_vae]

        cond["c_crossattn_1"]["null_image_emb"] = [fmri_null_x]
        cond["c_crossattn_1"]["null_text_emb"] = [fmri_null_cap]
        # import pdb; pdb.set_trace()
        cond["null_prompt_emb"] = [null_prompt.expand(len(cap),-1,-1)]

        # if self.is_fmri_input is True:
        #     cond["c_concat"] = [input_mask * self.fmri2visual_model((xc["c_concat"])).detach()]
        # else:
            # cond["c_concat"] = [input_mask * self.encode_first_stage((xc["c_concat"])).mode().detach()]
        c_concat = F.interpolate(xc["c_concat"], (sz,sz)) if sz is not None else xc["c_concat"]
        # c_concat = F.interpolate(xc["c_concat"], (256,256))
        cond["c_concat"] = [self.encode_first_stage(c_concat).mode().detach()]
        out = [z, cond]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out

    @torch.no_grad()
    def log_images(self, batch, epoch_n, iter_n, batch_idx, model_wrap, model_wrap_cfg,
                   save_dir, split,
                   cfg_text=7.5, cfg_fmri=1.5,
                   N=1, n_row=4, sample=True, 
                   steps=100, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=False):
        self.model.eval()
        s = batch['s'][0]

        N = min(batch['image'].shape[0], N)
        z_gt, c, x, xrec, xc = self.get_input(batch, self.first_stage_key, force_c_encode=True,
                                               bs=N, uncond=0, return_original_cond=True, 
                                               return_first_stage_outputs=True)
        # import pdb; pdb.set_trace();
        init_latent = torch.cat(c["c_crossattn_1"]["fmri_vae"],dim=0)

        self.device = z_gt.device
        self.sampler.model.model.diffusion_model.device = z_gt.device
        self.sampler.make_schedule(ddim_num_steps=self.ddim_steps, ddim_eta=self.ddim_eta, verbose=False)

        c_w_uncond = copy.deepcopy(c)

        # import pdb; pdb.set_trace();
        c0 = torch.cat(c["c_crossattn_1"]["image_emb"], 1)
        c1 = torch.cat(c["c_crossattn_1"]["text_emb"], 1)
        prompt_emb = torch.cat(c["c_crossattn"], 1)
        uncond_c0 = torch.cat(c["c_crossattn_1"]["null_image_emb"], 1)
        uncond_c1 = torch.cat(c["c_crossattn_1"]["null_text_emb"], 1)
        null_prompt_emb = torch.cat(c["null_prompt_emb"], 1)

        c_w_uncond["c_crossattn_1"]["image_emb"] = [torch.cat([uncond_c0, c0], 0)]
        c_w_uncond["c_crossattn_1"]["text_emb"] = [torch.cat([uncond_c1, c1], 0)]
        c_w_uncond["c_crossattn"] = [torch.cat([null_prompt_emb, prompt_emb], 0)]
        c_concat = F.interpolate(xc["c_concat"], (512, 512))
        c_concat = self.encode_first_stage(c_concat).mode().detach()
        c_w_uncond["c_concat"] = [torch.cat([c_concat,]*2, 0)]

        # z_enc = self.sampler.stochastic_encode(init_latent, torch.tensor([self.t_enc]).to(z_gt.device))
        # z_enc_gen = z_enc_edit = z_enc
        # self.t_enc = t_enc

        z_enc_edit = torch.randn_like(init_latent)
        z_enc_gen = torch.randn_like(init_latent)
        t_enc = 50

        instruct_cap = batch['fmri_edit']['c_crossattn'][0]
        ######### designed dual-stream diffusion sampling ##########
        z_gen, z_edit = self.sampler.decode_dual(
            x_latent_gen=z_enc_gen,
            x_latent_edit=z_enc_edit,
            t_start=t_enc,
            cond_dict=c_w_uncond,
            unconditional_guidance_scale_gen=cfg_text,
            unconditional_guidance_scale_edit=cfg_text,
            mixed_ratio=(1-self.mixing), 
        )
        x_gen = self.decode_first_stage(z_gen.half())
        x_edit = self.decode_first_stage(z_edit.half())
        # save_path = os.path.join("debug", "images", "new",  
        #         "all_iter-{:06}_ep-{:06}_bidx-{:06d}-{:06d}-{}.png".format(iter_n, epoch_n, batch_idx, s, instruct_cap))
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # torchvision.utils.save_image(x_edit*0.5+0.5, save_path)
        # return 

        # ######### another way to sampling ############
        # steps_std = 50
        # sigmas_std = model_wrap.get_sigmas(steps_std)
        # z_pred_std = torch.randn_like(z_enc)
        # # print('prompt_emb: ', prompt_emb.shape, 'null_prompt_emb:', null_prompt_emb.shape)
        # cond_std = {"c_crossattn": [prompt_emb], "c_concat": c['c_concat']}
        # uncond_std = {"c_crossattn": [null_prompt_emb], "c_concat": [torch.zeros_like(c['c_concat'][0])]}
        # extra_args = {
        #     "cond": cond_std,
        #     "uncond": uncond_std,
        #     "text_cfg_scale": cfg_text,
        #     "fmri_cfg_scale": 0.0,
        # }
        # z_pred_std = K.sampling.sample_euler_ancestral(model_wrap_cfg, z_pred_std, sigmas_std, extra_args=extra_args)
        # x_pred_std = self.decode_first_stage(z_pred_std)
        # save_path = os.path.join("debug", "images", "ori",  
        #         "all_iter-{:06}_ep-{:06}_bidx-{:06d}-{:06d}-{}.png".format(iter_n, epoch_n, batch_idx, s, instruct_cap))
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # torchvision.utils.save_image(x_pred_std*0.5+0.5, save_path)
        # return 
        # torchvision.utils.save_image(x_edit*0.5+0.5, 'x_edit.jpg')
        # torchvision.utils.save_image(x_gen*0.5+0.5, 'x_gen.jpg')

        # import pdb; pdb.set_trace()
        x_instruct_txt = log_txt_as_img((x_gen.shape[2], x_gen.shape[3]), xc["c_crossattn"])
        x_instruct_txt_resize = F.interpolate(x_instruct_txt, (x_gen.shape[2], x_gen.shape[3]))
        c_concat_resize = F.interpolate(xc["c_concat"], (x_gen.shape[2], x_gen.shape[3]))
        x_resize = F.interpolate(x, (x_gen.shape[2], x_gen.shape[3]))
        # import pdb; pdb.set_trace()
        x_cat = torch.cat([x_instruct_txt_resize.detach().cpu(), 
                            x_resize.detach().cpu(), c_concat_resize.detach().cpu(), 
                            x_gen.detach().cpu(), x_edit.detach().cpu()], dim=-2)
        x_cat = torch.clamp((x_cat+1.0)/2.0, min=0., max=1.)
        
        root = os.path.join(save_dir, "images", split)
        filename = "all_iter-{:06}_ep-{:06}_bidx-{:06d}-{:06d}.png".format(iter_n, epoch_n, batch_idx, s)
        path = os.path.join(root, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torchvision.utils.save_image(x_cat, path)

        self.model.train()
        # import pdb; pdb.set_trace()

    def apply_model(self, x_noisy_gen, x_noisy_edit, t, cond=None, return_ids=False):
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}
        
        if hasattr(self, "split_input_params"):
            assert len(cond) == 1  # todo can only deal with one conditioning atm
            assert not return_ids
            ks = self.split_input_params["ks"]  # eg. (128, 128)
            stride = self.split_input_params["stride"]  # eg. (64, 64)

            h, w = x_noisy.shape[-2:]

            fold, unfold, normalization, weighting = self.get_fold_unfold(x_noisy, ks, stride)

            z = unfold(x_noisy)  # (bn, nc * prod(**ks), L)
            # Reshape to img shape
            z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
            z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]

            if self.cond_stage_key in ["image", "LR_image", "segmentation",
                                       'bbox_img'] and self.model.conditioning_key:  # todo check for completeness
                c_key = next(iter(cond.keys()))  # get key
                c = next(iter(cond.values()))  # get value
                assert (len(c) == 1)  # todo extend to list with more than one elem
                c = c[0]  # get element

                c = unfold(c)
                c = c.view((c.shape[0], -1, ks[0], ks[1], c.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                cond_list = [{c_key: [c[:, :, :, :, i]]} for i in range(c.shape[-1])]

            elif self.cond_stage_key == 'coordinates_bbox':
                assert 'original_image_size' in self.split_input_params, 'BoudingBoxRescaling is missing original_image_size'

                # assuming padding of unfold is always 0 and its dilation is always 1
                n_patches_per_row = int((w - ks[0]) / stride[0] + 1)
                full_img_h, full_img_w = self.split_input_params['original_image_size']
                # as we are operating on latents, we need the factor from the original image size to the
                # spatial latent size to properly rescale the crops for regenerating the bbox annotations
                num_downs = self.first_stage_model.encoder.num_resolutions - 1
                rescale_latent = 2 ** (num_downs)

                # get top left postions of patches as conforming for the bbbox tokenizer, therefore we
                # need to rescale the tl patch coordinates to be in between (0,1)
                tl_patch_coordinates = [(rescale_latent * stride[0] * (patch_nr % n_patches_per_row) / full_img_w,
                                         rescale_latent * stride[1] * (patch_nr // n_patches_per_row) / full_img_h)
                                        for patch_nr in range(z.shape[-1])]

                # patch_limits are tl_coord, width and height coordinates as (x_tl, y_tl, h, w)
                patch_limits = [(x_tl, y_tl,
                                 rescale_latent * ks[0] / full_img_w,
                                 rescale_latent * ks[1] / full_img_h) for x_tl, y_tl in tl_patch_coordinates]
                # patch_values = [(np.arange(x_tl,min(x_tl+ks, 1.)),np.arange(y_tl,min(y_tl+ks, 1.))) for x_tl, y_tl in tl_patch_coordinates]

                # tokenize crop coordinates for the bounding boxes of the respective patches
                patch_limits_tknzd = [torch.LongTensor(self.bbox_tokenizer._crop_encoder(bbox))[None]
                                      for bbox in patch_limits]  # list of length l with tensors of shape (1, 2)
                print(patch_limits_tknzd[0].shape)
                # cut tknzd crop position from conditioning
                assert isinstance(cond, dict), 'cond must be dict to be fed into model'
                cut_cond = cond['c_crossattn'][0][..., :-2]
                print(cut_cond.shape)

                adapted_cond = torch.stack([torch.cat([cut_cond, p], dim=1) for p in patch_limits_tknzd])
                adapted_cond = rearrange(adapted_cond, 'l b n -> (l b) n')
                print(adapted_cond.shape)
                adapted_cond = self.get_learned_conditioning(adapted_cond)
                print(adapted_cond.shape)
                adapted_cond = rearrange(adapted_cond, '(l b) n d -> l b n d', l=z.shape[-1])
                print(adapted_cond.shape)

                cond_list = [{'c_crossattn': [e]} for e in adapted_cond]

            else:
                cond_list = [cond for i in range(z.shape[-1])]  # Todo make this more efficient

            # apply model by loop over crops
            output_list = [self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])]
            assert not isinstance(output_list[0],
                                  tuple)  # todo cant deal with multiple model outputs check this never happens

            o = torch.stack(output_list, axis=-1)
            o = o * weighting
            # Reverse reshape to img shape
            o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
            # stitch crops together
            x_recon = fold(o) / normalization

        else:
            ## only add this
            new_cond = copy.deepcopy(cond)
            new_cond["only_mid_control"] = self.only_mid_control
            c0 = torch.cat(new_cond["c_crossattn_1"]["image_emb"], 1)
            c1 = torch.cat(new_cond["c_crossattn_1"]["text_emb"], 1)
            fmri_vae = torch.cat(new_cond["c_crossattn_1"]["fmri_vae"],1)

            # import pdb; pdb.set_trace()
            # with torch.no_grad():
            x_recon_gen, control_res = self.control_model.forward_dc(x=torch.cat([x_noisy_gen], dim=1), 
                                                        # hint=fmri_vae,
                                                        timesteps=t,
                                                        c0=c0, c1=c1,
                                                        xtype='image', c0_type='vision', 
                                                        c1_type='prompt', mixed_ratio=0.6)
            # control_res = [tt.detach().requires_grad_(True) for tt in control_res]
            new_cond.pop('c_crossattn_1')
            new_cond.pop('null_prompt_emb')
            fmri_control = [c * scale for c, scale in zip(control_res, self.control_scales)]
            new_cond["control"] = fmri_control
            # new_cond["control"] = None
            ## only add above
            # import pdb; pdb.set_trace()
            x_recon_edit = self.model(x_noisy_edit, t, **new_cond)

        return x_recon_gen, x_recon_edit
