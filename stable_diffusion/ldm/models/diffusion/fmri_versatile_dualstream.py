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
from ldm.modules.encoders.modules import FrozenClipImageEmbedder
from ldm.util import default
from ldm.models.diffusion.alignblock import align_block


class ZeroConvControlledUnetModel(UNetModel):
    def __init__(self, train_feat_adaptor=False, train_res_inject_adaptor=False, conditioning_scale=1.0, *args, **kwargs):
        self.resnet_inject_add = kwargs.pop('resnet_inject_add', False)
        self.isTest = kwargs.pop('isTest', False)
        self.inject_min_step = kwargs.pop('inject_min_step', 500)

        super().__init__(*args, **kwargs)
        channel_mult = self.channel_mult
        num_res_blocks = self.num_res_blocks
        model_channels = self.model_channels
        attention_resolutions = self.attention_resolutions
        use_spatial_transformer = self.use_spatial_transformer
        num_head_channels = self.num_head_channels
        num_heads = self.num_heads
        legacy = self.legacy
        default_eps = self.default_eps
        context_dim = self.context_dim
        transformer_depth = self.transformer_depth
        force_type_convert = self.force_type_convert
        self.train_feat_adaptor = train_feat_adaptor
        self.train_res_inject_adaptor = train_res_inject_adaptor
        self.dims = 2

        assert not ((train_feat_adaptor is True) and (train_res_inject_adaptor is True)), 'train_feat_adaptor and train_res_inject_adaptor could only choose one.'

        if train_feat_adaptor is True:
            ds = 8
            layers = []
            out_ch = 1280
            cnt = 0
            self.adaptor_blocks = nn.ModuleList([])
            for level, mult in list(enumerate(channel_mult))[::-1]:
                for i in range(num_res_blocks + 1):
                    ch = mult * model_channels
                    if ds in attention_resolutions:
                        if num_head_channels == -1:
                            dim_head = ch // num_heads
                        else:
                            num_heads = ch // num_head_channels
                            dim_head = num_head_channels
                        if legacy:
                            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

                    if level and i == num_res_blocks:
                        out_ch = ch
                        ds //= 2

                    self.adaptor_blocks.append(
                        self.make_zero_conv(channels=ch, out_channels=ch))
                    cnt += 1
                    if cnt >=9: break
        
        if train_res_inject_adaptor is True:
            ds = 8
            layers = []
            out_ch = 1280
            cnt = 0
            self.adaptor_blocks = nn.ModuleList([])
            for level, mult in list(enumerate(channel_mult))[::-1]:
                for i in range(num_res_blocks + 1):
                    ch = mult * model_channels
                    if ds in attention_resolutions:
                        if num_head_channels == -1:
                            dim_head = ch // num_heads
                        else:
                            num_heads = ch // num_head_channels
                            dim_head = num_head_channels
                        if legacy:
                            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

                    if level and i == num_res_blocks:
                        out_ch = ch
                        ds //= 2

                    self.adaptor_blocks.append(
                        self.make_zero_conv(channels=ch, out_channels=ch))
                    cnt += 1
                    if cnt >=9: break
        
        scales = torch.logspace(0, -1, len(self.adaptor_blocks))  # 0.1 to 1.0
        scales = scales * conditioning_scale
        self.scales = scales
        # import pdb; pdb.set_trace()


    def make_zero_conv(self, channels, kernel_size=1, stride=1, out_channels=None):
        if out_channels is None:
            return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, kernel_size, stride=stride, padding=0)))
        else:
            return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, out_channels, kernel_size, stride=stride, padding=0)))

    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, 
                        num_control_layers=1000, injected_features=None, injected_contexts=None, 
                        injected_attn_qkv=None, is_return_x0=False, sqrt_one_minus_at=None, a_t=None, context_mask=None, **kwargs):
        if (control is not None and len(control)) or injected_features is not None or injected_attn_qkv is not None:
            if not self.isTest: context_mask = None
            # context_mask = None

            x0 = x.clone()[:,:4]
            hs = []
            with torch.no_grad():
                t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
                emb = self.time_embed(t_emb.type(self.time_embed[0].weight.dtype))
                h = x.type(self.dtype)
                for i, module in enumerate(self.input_blocks):
                    if context_mask is not None:
                        h = module(h, emb, context=context, context_mask=context_mask)
                    else:
                        h = module(h, emb, context)
                    hs.append(h)

                if context_mask is not None:
                    h = self.middle_block(h, emb, context=context, context_mask=context_mask)
                else:
                    h = self.middle_block(h, emb, context=context)

            if control is not None:
                h += control.pop(0)

            # import pdb; pdb.set_trace();

            # module_i = 0
            context_cnt, attn_cnt = 0, 0
            for i, module in enumerate(self.output_blocks):
                if only_mid_control or control is None or i > num_control_layers:# or i in unmatched_layers:
                    h = torch.cat([h, hs.pop()], dim=1)
                else:
                    h = torch.cat([h, hs.pop() + control.pop(0)], dim=1)

                injected_attn_q, injected_attn_k, injected_attn_v = None, None, None
                if injected_attn_qkv is not None and attn_cnt < len(injected_attn_qkv[0]):
                    injected_attn_q, injected_attn_k, injected_attn_v = \
                            injected_attn_qkv[0][attn_cnt], injected_attn_qkv[1][attn_cnt], injected_attn_qkv[2][attn_cnt]
                    attn_cnt += 1

                out_layers_feature_key = f'output_block_{i}_out_layers_features'
                if injected_features is not None and out_layers_feature_key in injected_features:
                    out_layers_injected = injected_features[out_layers_feature_key]
                    out_layers_injected_transformed = self.adaptor_blocks[i](out_layers_injected, emb) + out_layers_injected
                    # print(timesteps.shape, timesteps)
                    # import pdb; pdb.set_trace()
                    if timesteps[0].item() < self.inject_min_step and self.isTest:
                        h = module(h, emb, context, context_mask=context_mask)
                    else:
                        h = module(h, emb, context, out_layers_injected=out_layers_injected_transformed, context_mask=context_mask)

                else:
                    h = module(h, emb, context,
                                self_attn_k_injected=None, # injected_attn_k
                                self_attn_v_injected=None, # injected_attn_v
                                context_mask=context_mask)

                ### injected features have higher priority
                if injected_features is None and injected_contexts is not None and context_cnt < len(self.adaptor_blocks):
                    inject_context = injected_contexts[context_cnt]
                    inject_context = inject_context.detach().requires_grad_(True)
                    res_h = self.adaptor_blocks[i](inject_context, emb)
                    # print('res_h', torch.sum(torch.abs(res_h)), 'scale: ', self.scales[i])
                    h = h + res_h * self.scales[i]
                    context_cnt += 1
                    # print('i: ', i, 'h.shape: ', h.shape)
                
                # module_i += 1
                # print('controlled h: ', i, h.shape)

            # import pdb; pdb.set_trace()
            h = h.type(x.dtype)
            final_out = self.out(h)
            
            if is_return_x0 is False:
                return final_out
            else:
                # import pdb; pdb.set_trace()
                denoised_x0 = (x0 - sqrt_one_minus_at * final_out) / a_t.sqrt()
                denoised_x0_fix = denoised_x0 / 0.18215
                return final_out, denoised_x0_fix

        else:
            # import pdb; pdb.set_trace();
            return super().forward(x, timesteps=timesteps, context=context, **kwargs)


class PreVersatileNetAdaptor(UNetModelVD):
    def __init__(self, train_feat_adaptor=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dims = self.dims = 2
        model_channels = self.model_channels
        channel_mult = self.channel_mult
        self.train_feat_adaptor = train_feat_adaptor

        ch = channel_mult[-1] * model_channels
        self.middle_block_out = self.make_zero_conv(ch)

        # ### for post-feature extraction
        # self.zero_convs = nn.ModuleList([])
        # if self.train_feat_adaptor is True:
        #     for level_idx, mult in list(enumerate(channel_mult))[::-1]:
        #         for block_idx in range(self.num_noattn_blocks[level_idx] + 1):
        #             ch = mult * model_channels
        #             self.zero_convs.append(self.make_zero_conv(ch))

    def make_zero_conv(self, channels, kernel_size=1, stride=1, out_channels=None):
        if out_channels is None:
            return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, kernel_size, stride=stride, padding=0)))
        else:
            return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, out_channels, kernel_size, stride=stride, padding=0)))

    def forward_dc(self, x, timesteps, c0, c1, xtype, c0_type, c1_type, mixed_ratio, 
                    is_save_intermediate=True, is_save_x0=False, 
                    sqrt_one_minus_at=None, a_t=None):
        hs, outs = [], []
        # useful_block_idxes = [4, 5, 6, 7, 8] # this is too full
        useful_block_idxes = [4, 5, 6] # 
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        
        x=x.half()
        x0=x.clone()
        emb = self.time_embed(t_emb.half())

        if xtype == 'text':
            x = x[:, :, None, None]
        h = x

        # for i, (i_module, t_module, zero_conv) in enumerate(zip(self.unet_image.input_blocks, self.unet_text.input_blocks, self.zero_convs)):
        for i, (i_module, t_module) in enumerate(zip(self.unet_image.input_blocks, self.unet_text.input_blocks)):
            h = self.mixed_run_dc(i_module, t_module, h, emb, c0, c1, xtype, c0_type, c1_type, mixed_ratio)
            # if is_save_intermediate is True:
            #     outs.append(zero_conv(h, emb))
            hs.append(h)

        h = self.mixed_run_dc(self.unet_image.middle_block, self.unet_text.middle_block, 
                                h, emb, c0, c1, xtype, c0_type, c1_type, mixed_ratio)
        if is_save_intermediate is True:
            outs.append(self.middle_block_out(h, emb))

        block_idx = 0
        contexts = []
        attn_qs, attn_ks, attn_vs = [], [], []
        ## layer 2, 5, 8 include upsampling: (1280, 1280, 640)
        # for i, (i_module, t_module, zero_conv) in enumerate(zip(self.unet_image.output_blocks, self.unet_text.output_blocks, self.zero_convs)):
        for i, (i_module, t_module) in enumerate(zip(self.unet_image.output_blocks, self.unet_text.output_blocks)):
            # print('i: ', i)
            h = th.cat([h, hs.pop()], dim=1)
            h = self.mixed_run_dc(i_module, t_module, h, emb, c0, c1, xtype, c0_type, c1_type, mixed_ratio)
            if block_idx in useful_block_idxes:
                # outs.append(self.unet_image.output_blocks[block_idx][0].out_layers_features)
                feat_i = self.unet_image.output_blocks[block_idx][0].out_layers_features  ## resnet features
                attn_q = self.unet_image.output_blocks[block_idx][1].transformer_blocks[0].attn1.q
                attn_k = self.unet_image.output_blocks[block_idx][1].transformer_blocks[0].attn1.k
                attn_v = self.unet_image.output_blocks[block_idx][1].transformer_blocks[0].attn1.v
                # import pdb; pdb.set_trace()
                # if self.train_feat_adaptor is True:
                #     feat_i_transformed = zero_conv(feat_i, emb) + feat_i
                # else:
                feat_i_transformed = feat_i
                
                outs.append(feat_i_transformed)
                attn_qs.append(attn_q)
                attn_ks.append(attn_k)
                attn_vs.append(attn_v)

            block_idx += 1
            contexts.append(h)
            # print('pre extracted h: ', i, h.shape)

        attn_info = [attn_qs, attn_ks, attn_vs]
        final_out = self.unet_image.out(h)

        # import pdb; pdb.set_trace()
        if is_save_x0 is True:
            denoised_x0 = (x0 - sqrt_one_minus_at * final_out) / a_t.sqrt()
            denoised_x0_fix = denoised_x0 / 0.18215
            outs.append(denoised_x0_fix)
            # import pdb; pdb.set_trace();

        outs = list(reversed(outs))
        if xtype == 'image':
            return final_out, outs, contexts, attn_info
        elif xtype == 'text':
            return self.unet_text.out(h).squeeze(-1).squeeze(-1), outs, contexts


class DualLDM(LatentDiffusion):
    def __init__(self, clip_cfg, fmri2clip_cfg=None, fmri2clip_pretrain_path=None,
                fmri_vclip_cfg=None, fmri_vclip_pretrain_path=None, personalization_config=None, *args, **kwargs):
        
        self.use_resnet_inject = kwargs.pop('use_resnet_inject', False)
        self.is_no_inject = kwargs.pop('is_no_inject', False)
        self.is_no_concat = kwargs.pop('is_no_concat', False)

        super().__init__(*args, **kwargs)
        self.vd_clip = VDCLIP(clip_cfg)

        self.adaptor_ckpt_path = kwargs.get('adaptor_ckpt_path', None)

        self.useful_ctx_idxes = kwargs.get('useful_ctx_idxes', None)
        self.mutual_selfatt = kwargs.get('mutual_selfatt', None)
        self.only_align_loss = kwargs.get('only_align_loss', False)
        self.use_fmri_clip = kwargs.get('use_fmri_clip', False)
        self.use_fmri_vae = kwargs.get('use_fmri_vae', True)


        self.use_styleclip_loss = kwargs['use_styleclip_loss']
        self.is_save_x0 = kwargs['is_save_x0']
        self.is_save_intermediate = kwargs['is_save_intermediate']
        self.coarse_spatial_steps = kwargs['coarse_spatial_steps']
        pretrained_control_unet_path = kwargs['pretrained_control_unet_path']
        if pretrained_control_unet_path is not None and os.path.exists(pretrained_control_unet_path):
            # import pdb; pdb.set_trace()
            pretrained_state_dict = torch.load(pretrained_control_unet_path, map_location="cpu")
            pretrained_state_dict = {k:v for k,v in pretrained_state_dict.items() if 'clip.' in k}
            missing, unexpected = self.vd_clip.load_state_dict(pretrained_state_dict, strict=False)

            print('versatile clip missing {} params.'.format(len(missing)))
            print('versatile clip missing: ', missing)
            print('versatile clip unexpected {} params.'.format(len(unexpected)))
        
        ctx_adaptor_ckpt_path = kwargs.get('ctx_adaptor_ckpt_path', None)
        # ctx_adaptor_ckpt_path = kwargs['ctx_adaptor_ckpt_path']
        if ctx_adaptor_ckpt_path is not None and os.path.exists(ctx_adaptor_ckpt_path):
            pretrained_state_dict = torch.load(ctx_adaptor_ckpt_path, map_location="cpu")
            # import pdb; pdb.set_trace()
            ctx_pretrained_state_dict = {k.replace('model.diffusion_model.adaptor_blocks.',''): v for k,v in pretrained_state_dict['module'].items() if 'model.diffusion_model.adaptor_blocks.' in k}
            missing, unexpected = self.model.diffusion_model.adaptor_blocks.load_state_dict(ctx_pretrained_state_dict, strict=False)
            print('ctx missing: ', len(missing))

        self.prospect_stages = 10
        self.sampler = DDIMSampler_Dual(self)

        # ddim_steps = 50
        ddim_steps = 50
        ddim_eta = 0
        scale = 7.5
        strength = 0.75

        mixing = kwargs.get('mixing', 0.4) ### mix of image an text
        t_enc = int(strength * ddim_steps)

        self.t_enc = t_enc
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.scale = scale
        self.mixing = mixing

        # import pdb; pdb.set_trace();
        self.fmri2clip_model = None
        if fmri2clip_cfg is not None:
            self.fmri2clip_pretrain_path = fmri2clip_pretrain_path
            self.fmri2clip_model = self.instantiate_fmri2clip(fmri2clip_cfg)
            self.fmri2clip_model.eval()

        if fmri_vclip_cfg is not None:
            self.fmri_vclip_pretrain_path = fmri_vclip_pretrain_path
            self.instantiate_fmri_vclip(fmri_vclip_cfg)

        if self.use_styleclip_loss is True:
            from ldm.modules.losses.clip_loss import CLIPLoss
            self.styleclip_loss = CLIPLoss()

        if self.only_align_loss is False:
            self.image_clip = FrozenClipImageEmbedder()

        self.embedding_manager = None
        if personalization_config:
            self.embedding_manager = self.instantiate_embedding_manager(personalization_config, self.cond_stage_model)
        if self.embedding_manager:
            for param in self.embedding_manager.embedding_parameters():
                param.requires_grad = False
        self.device = next(self.parameters()).device

    def instantiate_fmri2clip(self, config):
        model = instantiate_from_config(config)
        if self.fmri2clip_pretrain_path and os.path.exists(self.fmri2clip_pretrain_path):
            # import pdb; pdb.set_trace()
            state_dict = torch.load(self.fmri2clip_pretrain_path)['module']
            filter_state_dict = {k.replace('fmri2clip_model.',''):v for k,v in state_dict.items() if 'fmri2clip_model.' in k}
            missing, unexpected = model.load_state_dict(filter_state_dict, strict=False)
            print('missing: ', missing)
            print('unexpected: ', unexpected)
            # import pdb; pdb.set_trace()
        else:
            print('fmri2clip_pretrain_path not exist.')
        return model

    def instantiate_embedding_manager(self, config, embedder):
        model = instantiate_from_config(config, embedder=embedder)

        if config.params.get("embedding_manager_ckpt", None): # do not load if missing OR empty string
            model.load(config.params.embedding_manager_ckpt)
        
        return model

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
    


    def get_clip_loss(self, pred_image, gt_image):
        pred_image_token = self.image_clip(pred_image.half())
        gt_image_token = self.image_clip(gt_image.half())
        
        pred_image_emb = F.normalize(pred_image_token, p=2, dim=1)
        gt_image_emb = F.normalize(gt_image_token, p=2, dim=1)

        cosine_sim = F.cosine_similarity(pred_image_emb, gt_image_emb)
        cosine_loss = 1 - cosine_sim

        l1_loss = nn.L1Loss()(pred_image_emb, gt_image_emb)
        return cosine_loss, l1_loss

    ### TODO: what we should give to noise_edit
    def p_losses(self, x_start_gen, x_start_edit, cond, t, input_text, output_text, edit_text, noise=None, noise_edit=None, t_edit=None, is_return_x0=True):

        noise_gen = default(noise, lambda: torch.randn_like(x_start_gen))
        x_noisy_gen = self.q_sample(x_start=x_start_gen, t=t, noise=noise_gen)

        noise_edit = default(noise_edit, lambda: torch.randn_like(x_start_edit))
        noise = noise_edit

        t_edit = t.clone() if t_edit is None else t_edit
        x_noisy_edit = self.q_sample(x_start=x_start_edit, t=t_edit, noise=noise_edit)
        # import pdb; pdb.set_trace();

        b = x_start_gen.shape[0]
        extended_shape = (b, 1, 1, 1)
        alphas = self.alphas_cumprod #if use_original_steps else self.ddim_alphas
        a_t = torch.full(extended_shape, 1., device=x_noisy_gen.device, dtype=x_noisy_gen.dtype)
        for kk in range(b): a_t[kk] = alphas[t[kk]]

        sqrt_one_minus_alphas = self.sqrt_one_minus_alphas_cumprod
        sqrt_one_minus_at = torch.full(extended_shape, 1., device=x_noisy_gen.device, dtype=x_noisy_gen.dtype)
        for kk in range(b): sqrt_one_minus_at[kk] = sqrt_one_minus_alphas[t[kk]]

        # import pdb; pdb.set_trace();
        # print(t_edit.shape[0])
        offset = (t_edit[0].item()-t[0].item())//20
        a_t_offset = torch.full(extended_shape, 1., device=x_noisy_gen.device, dtype=x_noisy_gen.dtype)
        for kk in range(b): a_t_offset[kk] = alphas[t[kk]+offset*20]

        sqrt_one_minus_at_offset = torch.full(extended_shape, 1., device=x_noisy_gen.device, dtype=x_noisy_gen.dtype)
        for kk in range(b): sqrt_one_minus_at_offset[kk] = sqrt_one_minus_alphas[t[kk]+offset*20]
        
        # print('batch:  {}'.format(b))
        # import pdb; pdb.set_trace()
        noisy_c_concat4train = x_start_gen.clone() #/ 0.18215 ## theoretically speaking, should be the first add noise 15 steps and use first stream network to denoise.
        if is_return_x0 is False:
            model_output_gen, model_output = self.apply_model(x_noisy_gen, x_noisy_edit, t, cond, t_edit_in=t_edit, 
                            is_save_x0=self.is_save_x0, is_save_intermediate=self.is_save_intermediate,
                            sqrt_one_minus_at=sqrt_one_minus_at, a_t=a_t,
                            sqrt_one_minus_at_offset=sqrt_one_minus_at_offset, 
                            a_t_offset=a_t_offset,
                            noisy_c_concat4train=noisy_c_concat4train)
        else:
            model_output_gen, model_output, model_output_x0 = self.apply_model(x_noisy_gen, x_noisy_edit, t, cond, t_edit_in=t_edit, 
                            is_save_x0=self.is_save_x0, is_save_intermediate=self.is_save_intermediate,
                            sqrt_one_minus_at=sqrt_one_minus_at, a_t=a_t, 
                            sqrt_one_minus_at_offset=sqrt_one_minus_at_offset, 
                            a_t_offset=a_t_offset, 
                            is_return_x0=is_return_x0,
                            noisy_c_concat4train=noisy_c_concat4train)

            model_output_gen_x0 = (x_noisy_gen - sqrt_one_minus_at * model_output_gen) / a_t.sqrt()
            model_output_gen_x0 = model_output_gen_x0 / 0.18215
        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        
        # import pdb; pdb.set_trace();
        if self.parameterization == "x0":
            target = x_start_edit
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        # import pdb; pdb.set_trace()
        if is_return_x0 is False:
            loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        else:

            self.first_stage_model = self.first_stage_model.float()
            pred_image_x0 = self.differentiable_decode_first_stage(model_output_x0*0.18215)
            # gen_image_x0 = self.decode_first_stage(model_output_gen_x0*0.18215)
            gt_image_x0 = self.decode_first_stage(x_start_edit).detach().requires_grad_(True)
            # print(pred_image_x0.dtype, gt_image_x0.dtype)
            # import pdb; pdb.set_trace();
            # pred_gt = torch.cat([pred_image_x0, gt_image_x0, gen_image_x0], dim=2)
            # torchvision.utils.save_image(pred_gt*0.5+0.5, 'pred_gt_cat3.jpg')
            # import pdb; pdb.set_trace()

            # loss_cosine, loss_l1 = self.get_clip_loss(pred_image_x0, gt_image_x0)
            loss_simple_ori = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
            # loss_simple = loss_cosine + loss_l1 + loss_styleclip
            # loss_simple = loss_l1 + loss_simple_ori
            loss_simple = loss_simple_ori
            if self.use_styleclip_loss is True:
                # print(t, t_edit)
                image_start_gen = self.decode_first_stage(x_start_gen).detach()
                loss_styleclip = self.styleclip_loss(image_start_gen, pred_image_x0, input_text, output_text) #* 0.1
                loss_simple = loss_simple + loss_styleclip
    
            # loss_simple = loss_styleclip
            # loss_simple = loss_cosine
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})
        # loss_dict.update({f'{prefix}/loss_simple_cosine': loss_cosine.mean()})
        # loss_dict.update({f'{prefix}/loss_simple_l1': loss_l1.mean()})
        # loss_dict.update({f'{prefix}/loss_simple_styleclip': loss_styleclip.mean()})

        # additional_loss_type is in the format of min_snr_k
        if self.additional_loss_type is not None and isinstance(self.additional_loss_type, str) and self.additional_loss_type.startswith("min_snr_"):
            import pdb; pdb.set_trace();
            k = float(self.additional_loss_type.split("_")[-1])
            alpha = extract_into_tensor(self.sqrt_alphas_cumprod, t, t.shape)
            sigma = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, t.shape)

            snr = (alpha / sigma) ** 2
            min_snr = torch.stack([snr, k * torch.ones_like(t)], dim=1).min(dim=1)[0]
            if self.parameterization == "eps":
                loss_simple = loss_simple * min_snr / snr
            elif self.parameterization == "x0":
                loss_simple = loss_simple * min_snr
            else:
                raise NotImplementedError()

            loss_simple = loss_simple * min_snr

        if t_edit is not None:
            logvar_t_edit = self.logvar.to(x_start_edit.device)[t_edit]
            loss = loss_simple / torch.exp(logvar_t_edit) + logvar_t_edit
        else:
            logvar_t = self.logvar.to(x_start_edit.device)[t]
            loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        if is_return_x0 is False:
            loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        else:
            # loss_cosine_vlb, loss_l1_vlb = self.get_clip_loss(pred_image_x0, gt_image_x0)
            
            # loss_vlb = loss_cosine_vlb + loss_l1_vlb + loss_styleclip_vlb
            loss_vlb_ori = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
            # loss_vlb = loss_l1_vlb + loss_vlb_ori
            loss_vlb = loss_vlb_ori
            if self.use_styleclip_loss is True:
                loss_styleclip_vlb = self.styleclip_loss(image_start_gen, pred_image_x0, input_text, output_text) #* 0.1
                loss_vlb = loss_vlb + loss_styleclip_vlb

            # loss_vlb = loss_styleclip_vlb

            # loss_vlb = self.get_loss(model_output_x0, x_start_edit, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})
        # import pdb; pdb.set_trace();
        return loss, loss_dict

    def forward(self, batch, batch_idx, num_steps, *args, **kwargs):
        if self.only_align_loss is True:
            x, c = self.get_input(batch, self.first_stage_key, force_c_encode=True)
            loss, loss_dict = self.align_losses(c, *args, **kwargs)
        else:
            x, c = self.get_input(batch, self.first_stage_key)
            ratio = self.num_timesteps // self.ddim_steps
            t = torch.randint(0, self.num_timesteps-self.coarse_spatial_steps*ratio, (x.shape[0],), device=x.device).long()
            if self.model.conditioning_key is not None:
                assert c is not None
                if self.cond_stage_trainable:
                    c = self.get_learned_conditioning(c)
                if self.shorten_cond_schedule:  # TODO: drop this option
                    tc = self.cond_ids[t]
                    c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
            # import pdb; pdb.set_trace();
            input_text = batch['cap']
            output_text = batch['fmri_edit']['output']
            edit_text = batch['fmri_edit']['c_crossattn']
            loss, loss_dict = self.p_losses(c['c_concat'][0]*0.18215, x, c, t, input_text, output_text, edit_text, t_edit=t.clone()+self.coarse_spatial_steps*ratio, *args, **kwargs)

        return loss, loss_dict
    
    def align_losses(self, cond):
        # import pdb; pdb.set_trace();
        gt_image_emb = cond["c_crossattn_1"]["gt_image_emb"][0]
        gt_text_emb = cond["c_crossattn_1"]["gt_text_emb"][0]
        pred_image_emb = cond["c_crossattn_1"]["image_emb"][0]
        pred_text_emb = cond["c_crossattn_1"]["text_emb"][0]

        loss = F.mse_loss(pred_image_emb, gt_image_emb) + F.mse_loss(pred_text_emb, gt_text_emb)
        # loss = F.mse_loss(pred_image_emb, gt_image_emb) #+ F.mse_loss(pred_text_emb, gt_text_emb)
        loss_dict = {}
        loss_dict.update({'loss_align': loss})

        return loss, loss_dict

    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None, uncond=0.075, sz=256,
                  prospect_words=None):
        x = DDPM.get_input(self, batch, k)
        # import pdb; pdb.set_trace();
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

        pred_image_emb, pred_text_emb = None, None
        if self.fmri2clip_model is not None:
            if self.only_align_loss is True:
                self.fmri2clip_model.train()
                voxel = batch['fmri'].to(z)
                if bs is not None: voxel = voxel[:bs]
                if voxel.shape[1] == 3: voxel = voxel.mean(dim=1)
                self.fmri2clip_model = self.fmri2clip_model.float()
                pred_image_emb, pred_text_emb = self.fmri2clip_model(voxel)
                # pred_image_emb = self.fmri2clip_model(voxel)
            else:
                with torch.no_grad():
                    voxel = batch['fmri'].to(z)
                    if bs is not None: voxel = voxel[:bs]
                    if voxel.shape[1] == 3: voxel = voxel.mean(dim=1)
                    self.fmri2clip_model = self.fmri2clip_model.float()
                    pred_image_emb, pred_text_emb = self.fmri2clip_model(voxel)
                    # pred_image_emb = self.fmri2clip_model(voxel)
                    # import pdb; pdb.set_trace()
            # import pdb; pdb.set_trace();
            pred_image_emb = pred_image_emb.reshape(pred_image_emb.shape[0], -1, 768)
            pred_text_emb = pred_text_emb.reshape(pred_text_emb.shape[0], -1, 768)
            
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
        
        fmri_null_x = self.vd_clip.clip_encode_vision(null_x)
        fmri_null_cap = self.vd_clip.clip_encode_text(null_cap)
        fmri_x = self.vd_clip.clip_encode_vision(xc["c_concat"])
        fmri_cap = self.vd_clip.clip_encode_text(cap)

        cond["c_crossattn_1"] = {}

        if self.only_align_loss is True:
            cond["c_crossattn_1"]["gt_image_emb"] = [fmri_x.float().detach().requires_grad_(True)]
            cond["c_crossattn_1"]["gt_text_emb"] = [fmri_cap.float().detach().requires_grad_(True)]

        if pred_image_emb is not None:
            if self.only_align_loss is True:
                fmri_x, fmri_cap = pred_image_emb, pred_text_emb
                # fmri_x = pred_image_emb
            else:
                fmri_x, fmri_cap = pred_image_emb.half(), pred_text_emb.half()
                # fmri_x = pred_image_emb.half()

        # if 'img_clip' in batch:
        #     # import pdb; pdb.set_trace();
        #     batch['img_clip'] = batch['img_clip'].to(z)[:bs]
        #     print(batch['img_clip'].shape, fmri_x.shape, torch.nn.MSELoss()(batch['img_clip'], fmri_x))
        #     fmri_x = batch['img_clip'].half()

        # import pdb; pdb.set_trace();
        if force_c_encode is False:
            cond["c_crossattn_1"]["image_emb"] = [torch.where(fmri_prompt_mask.bool(), fmri_null_x, fmri_x)]
            cond["c_crossattn_1"]["text_emb"] = [torch.where(fmri_prompt_mask.bool(), fmri_null_cap, fmri_cap)]
            cond["c_crossattn"] = [torch.where(prompt_mask, null_prompt[0].detach(), self.get_learned_conditioning(xc["c_crossattn"])[0].detach())]
        else:
            cond["c_crossattn_1"]["image_emb"] = [fmri_x]
            cond["c_crossattn_1"]["text_emb"] = [fmri_cap]
            if self.embedding_manager is not None:
                if prospect_words is not None:
                    cond["c_crossattn"] = [self.get_learned_conditioning(['*'], prospect_words=prospect_words)[0].detach()]
                else:
                    cond["c_crossattn"] = [self.get_learned_conditioning(xc["c_crossattn"])[0].detach()]
            else:
                cond["c_crossattn"] = [self.get_learned_conditioning(xc["c_crossattn"]).detach()]

        cond["c_crossattn_1"]["fmri_vae"] = [fmri_vae]

        cond["c_crossattn_1"]["null_image_emb"] = [fmri_null_x]
        cond["c_crossattn_1"]["null_text_emb"] = [fmri_null_cap]
        cond["null_prompt_emb"] = [null_prompt[0].expand(len(cap),-1,-1)]

        c_concat = F.interpolate(xc["c_concat"], (sz,sz)) if sz is not None else xc["c_concat"]
        # import pdb; pdb.set_trace()
        # print(c_concat.shape, c_concat.max(), c_concat.min())
        cond["c_concat"] = [self.encode_first_stage(c_concat).mode().detach()]
        out = [z, cond]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out

    @torch.no_grad()
    def save_text(self, batch, epoch_n, iter_n, batch_idx, model_wrap, model_wrap_cfg,
                   save_dir, split,
                   cfg_text=7.5, cfg_fmri=1.5,
                   cfg_text_edit=None, cfg_image_edit=None,
                   delay_t=None,
                   N=1, n_row=4, sample=True, 
                   steps=100, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=False,
                   prospect_words=None, 
                   is_inst_edit=False,
                   is_inst_gen=False,
                   layout_in=None):
        s = batch['s'][0]
        root = os.path.join(save_dir, "images", split)
        output_filename = "all_iter-{:06}_ep-{:06}_bidx-{:06d}-{:06d}-output.txt".format(iter_n, epoch_n, batch_idx, s)
        input_filename = "all_iter-{:06}_ep-{:06}_bidx-{:06d}-{:06d}-input.txt".format(iter_n, epoch_n, batch_idx, s)
        instruct_filename = "all_iter-{:06}_ep-{:06}_bidx-{:06d}-{:06d}-instruct.txt".format(iter_n, epoch_n, batch_idx, s)
        output_path = os.path.join(root, output_filename)
        input_path = os.path.join(root, input_filename)
        instruct_path = os.path.join(root, instruct_filename)
        os.makedirs(os.path.dirname(input_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs(os.path.dirname(instruct_path), exist_ok=True)

        input_cap = batch['cap'][0]
        output_cap = batch['fmri_edit']['output'][0]
        instruct_cap = batch['fmri_edit']['c_crossattn'][0]

        with open(input_path, "w") as file:
            file.write(input_cap)
        with open(output_path, "w") as file:
            file.write(output_cap)
        with open(instruct_path, "w") as file:
            file.write(instruct_cap)

    @torch.no_grad()
    def save_fmri_vae(self, batch, epoch_n, iter_n, batch_idx, model_wrap, model_wrap_cfg,
                   save_dir, split,
                   cfg_text=7.5, cfg_fmri=1.5,
                   cfg_text_edit=None, cfg_image_edit=None,
                   delay_t=None,
                   N=1, n_row=4, sample=True, 
                   steps=100, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=False,
                   prospect_words=None, 
                   is_inst_edit=False,
                   is_inst_gen=False,
                   layout_in=None):
        if self.model is None: return
        self.model.eval()
        s = batch['s'][0]
        root = os.path.join(save_dir, "images", split)

        N = min(batch['image'].shape[0], N)
        z_gt, c, x, xrec, xc = self.get_input(batch, self.first_stage_key, force_c_encode=True,
                                               bs=N, uncond=0, return_original_cond=True, 
                                               return_first_stage_outputs=True,
                                               prospect_words=prospect_words)
        
        if layout_in is not None:
            sz = 512
            layout_in = F.interpolate(layout_in, (sz,sz))
            layout_concat = self.encode_first_stage(layout_in).mode().detach()

        init_latent = torch.cat(c["c_crossattn_1"]["fmri_vae"],dim=0)
        output_basename = '{:05d}.npy'.format(s)
        output_path = os.path.join(root, output_basename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # import pdb; pdb.set_trace()
        np.save(output_path, init_latent.cpu().numpy()[0])

    @torch.no_grad()
    def log_images(self, batch, epoch_n, iter_n, batch_idx, model_wrap, model_wrap_cfg,
                   save_dir, split,
                   cfg_text=7.5, cfg_fmri=1.5,
                   cfg_text_edit=None, cfg_image_edit=None,
                   delay_t=None,
                   N=1, n_row=4, sample=True, 
                   steps=100, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=False,
                   prospect_words=None, 
                   is_inst_edit=False,
                   is_inst_gen=False,
                   layout_in=None):
        if self.model is None: return
        self.model.eval()
        s = batch['s'][0]

        N = min(batch['image'].shape[0], N)
        z_gt, c, x, xrec, xc = self.get_input(batch, self.first_stage_key, force_c_encode=True,
                                               bs=N, uncond=0, return_original_cond=True, 
                                               return_first_stage_outputs=True,
                                               prospect_words=prospect_words)
        
        if layout_in is not None:
            sz = 512
            layout_in = F.interpolate(layout_in, (sz,sz))
            # print(layout_in.max(), layout_in.min())
            layout_concat = self.encode_first_stage(layout_in).mode().detach()

        # import pdb; pdb.set_trace();
        if self.use_fmri_vae and split == 'val':
            init_latent = batch['fmri_vae'][:N].to(z_gt)
        else:
            init_latent = torch.cat(c["c_crossattn_1"]["fmri_vae"],dim=0)

        self.device = z_gt.device
        self.sampler.model.model.diffusion_model.device = z_gt.device
        self.sampler.make_schedule(ddim_num_steps=self.ddim_steps, ddim_eta=self.ddim_eta, verbose=False)

        z_enc_gen = self.sampler.stochastic_encode(init_latent, torch.tensor([int(0.75*self.ddim_steps)]).to(z_gt.device))
        z_enc_edit = torch.randn_like(init_latent)
        
        c_w_uncond = copy.deepcopy(c)

        # import pdb; pdb.set_trace();

        if self.use_fmri_clip and split == 'val':
            c0 = batch['img_clip'][:N].to(c["c_crossattn_1"]["image_emb"][0])
            c1 = batch['text_clip'][:N].to(c["c_crossattn_1"]["text_emb"][0])
            ## we borrow this logic and also write context_mask here
            context_mask = batch['context_mask'][:N].to(c["c_crossattn_1"]["image_emb"][0])
            context_mask = context_mask / 255.0
            context_mask = context_mask.unsqueeze(1)
            context_mask = F.interpolate(context_mask, (256, 256))#.flatten(0).unsqueeze(0).flatten()
            # import pdb; pdb.set_trace();
            c_w_uncond['context_mask'] = torch.cat([context_mask, context_mask, context_mask], 0)

        else:
            c0 = torch.cat(c["c_crossattn_1"]["image_emb"], 1)
            c1 = torch.cat(c["c_crossattn_1"]["text_emb"], 1)
            
            # context_mask = torch.zeros((N,1,16,16)).to(c0) 
            # context_mask = F.interpolate(context_mask, (16, 16)).flatten(0).unsqueeze(0)
            # context_mask = context_mask.flatten()
            # c_w_uncond['context_mask'] = torch.cat([context_mask, context_mask, context_mask], 0)
            c_w_uncond['context_mask'] = None

        # if is_inst_gen is True:
        #     # self.mixing = 1.0 # all text
        #     self.mixing = 0.0 # all image
        #     # c1 = [self.get_learned_conditioning(['*'],prospect_words=['*']*10)[0].detach()]
        #     # c1 = self.get_learned_conditioning(['A bird'])[0].detach()
        #     # c1 = self.vd_clip.clip_encode_text(['A cat.'])
        #     c1 = torch.cat(c["c_crossattn_1"]["text_emb"], 1)
        #     # import pdb; pdb.set_trace();
        #     # c1 = torch.cat(c1, 1)
        #     z_enc_gen = torch.randn_like(z_enc_gen)
        #     # c1 = torch.cat(c["c_crossattn_1"]["text_emb"], 1)
        #     # import pdb; pdb.set_trace();
        # else:
        #     # cond["c_crossattn"] = [self.get_learned_conditioning(['*'], prospect_words=prospect_words)[0].detach()]
        #     c1 = torch.cat(c["c_crossattn_1"]["text_emb"], 1)

        prompt_emb = torch.cat(c["c_crossattn"], 1)
        uncond_c0 = torch.cat(c["c_crossattn_1"]["null_image_emb"], 1)
        uncond_c1 = torch.cat(c["c_crossattn_1"]["null_text_emb"], 1)
        null_prompt_emb = torch.cat(c["null_prompt_emb"], 1)

        # cfg_text_edit, cfg_image_edit = 4.5, 1.5
        cfg_text_edit = 4.5 if cfg_text_edit is None else cfg_text_edit
        cfg_image_edit = 1.5 if cfg_image_edit is None else cfg_image_edit
        unconditional_guidance_scale_edit = None

        # context_mask = torch.zeros((N,1,256,256)).to(c0)
        if cfg_text_edit is not None and cfg_image_edit is not None:
            c_w_uncond["c_crossattn_1"]["image_emb"] = [torch.cat([uncond_c0, c0, c0], 0)]
            if is_inst_gen is True:
                new_c1 = torch.cat(c["c_crossattn_1"]["text_emb"], 1)
                c_w_uncond["c_crossattn_1"]["text_emb"] = [torch.cat([uncond_c1, new_c1, new_c1], 0)]*7 + [torch.cat([uncond_c1, c1, c1], 0)]*3
            else:
                c_w_uncond["c_crossattn_1"]["text_emb"] = [torch.cat([uncond_c1, c1, c1], 0)]
            # import pdb; pdb.set_trace();
            c_w_uncond["c_crossattn"] = [torch.cat([null_prompt_emb, prompt_emb, prompt_emb], 0)]
            if layout_in is None:
                c_concat = F.interpolate(xc["c_concat"], (512, 512))
                c_concat = self.encode_first_stage(c_concat).mode().detach()
                c_w_uncond["c_concat"] = [torch.cat([c_concat, torch.zeros_like(c_concat), c_concat], 0)]
            else:
                # c_w_uncond["c_crossattn"] = [torch.cat([null_prompt_emb, null_prompt_emb, null_prompt_emb], 0)]
                c_w_uncond["c_crossattn"] = [torch.cat([null_prompt_emb, c1, c1], 0)]
                c_w_uncond["layout_concat"] = [torch.cat([layout_concat, torch.zeros_like(layout_concat), layout_concat], 0)]
        else:
            c_w_uncond["c_crossattn_1"]["image_emb"] = [torch.cat([uncond_c0, c0], 0)]
            c_w_uncond["c_crossattn_1"]["text_emb"] = [torch.cat([uncond_c1, c1], 0)]
            c_w_uncond["c_crossattn"] = [torch.cat([null_prompt_emb, prompt_emb], 0)]
            if layout_in is None:
                c_concat = F.interpolate(xc["c_concat"], (512, 512))
                c_concat = self.encode_first_stage(c_concat).mode().detach()
                c_w_uncond["c_concat"] = [torch.cat([c_concat, torch.zeros_like(c_concat), c_concat], 0)]
            else:
                c_w_uncond["c_crossattn"] = [torch.cat([null_prompt_emb, null_prompt_emb], 0)]
                c_w_uncond["layout_concat"] = [torch.cat([layout_concat, torch.zeros_like(layout_concat), layout_concat], 0)]
        
        t_enc = 49
        instruct_cap = batch['fmri_edit']['c_crossattn'][0]
        ######### designed dual-stream diffusion sampling ##########
        # import pdb; pdb.set_trace()
        if unconditional_guidance_scale_edit is not None:
            z_gen, z_edit = self.sampler.decode_dual(
                x_latent_gen=z_enc_gen,
                x_latent_edit=z_enc_edit,
                t_start=t_enc,
                cond_dict=c_w_uncond,
                unconditional_guidance_scale_gen=cfg_text,
                unconditional_guidance_scale_edit=unconditional_guidance_scale_edit,
                unconditional_guidance_scale_text_edit=None,
                unconditional_guidance_scale_image_edit=None,
                mixed_ratio=(1-self.mixing), 
                is_inst_edit=is_inst_edit,
                coarse_spatial_steps=self.coarse_spatial_steps
            )
        else:
            z_gen, z_edit, z0_gen_info, z0_edit_info = self.sampler.decode_dual(
                x_latent_gen=z_enc_gen,
                x_latent_edit=z_enc_edit,
                t_start=t_enc,
                cond_dict=c_w_uncond,
                unconditional_guidance_scale_gen=cfg_text,
                unconditional_guidance_scale_edit=None,
                unconditional_guidance_scale_text_edit=cfg_text_edit,
                unconditional_guidance_scale_image_edit=cfg_image_edit,
                delay_t=delay_t,
                mixed_ratio=(1-self.mixing),
                is_save_intermediate=self.is_save_intermediate,
                is_save_x0=self.is_save_x0,
                is_inst_edit=is_inst_edit,
                coarse_spatial_steps=self.coarse_spatial_steps
            )
            # import pdb; pdb.set_trace()
            x0_gen_info, x0_edit_info = [], []
            for z0_gen, z0_edit in zip(z0_gen_info, z0_edit_info):
                x0_gen = self.decode_first_stage(z0_gen)
                x0_edit = self.decode_first_stage(z0_edit)
                x0_gen_info.append(x0_gen)
                x0_edit_info.append(x0_edit)
            # import pdb; pdb.set_trace()
            x0_gen_info = torch.cat(x0_gen_info, dim=0)
            x0_edit_info = torch.cat(x0_edit_info, dim=0)
            x0_info = torch.cat([x0_gen_info, x0_edit_info], dim=-2)*0.5+0.5

        x_gen = self.decode_first_stage(z_gen.half())
        x_edit = self.decode_first_stage(z_edit.half())

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

        filename_inter = "intermediate_all_iter-{:06}_ep-{:06}_bidx-{:06d}-{:06d}.png".format(iter_n, epoch_n, batch_idx, s)
        path_inter = os.path.join(root, filename_inter)
        os.makedirs(os.path.dirname(path_inter), exist_ok=True)
        torchvision.utils.save_image(x0_info, path_inter)

        x_gen_filename = '{:06d}.png'.format(s)
        x_gen_path = os.path.join(root, x_gen_filename)
        os.makedirs(os.path.dirname(x_gen_path), exist_ok=True)
        torchvision.utils.save_image(x_gen*0.5+0.5, x_gen_path)
        self.model.train()
        # import pdb; pdb.set_trace()

    def apply_model(self, x_noisy_gen, x_noisy_edit, t, cond=None, t_edit_in=None, is_save_intermediate=True, is_save_x0=False, 
                        is_return_x0=False, sqrt_one_minus_at=None, a_t=None, sqrt_one_minus_at_offset=None, a_t_offset=None, 
                        noisy_c_concat4train=None,
                        return_ids=False, mixed_ratio=0.6):
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
            # new_cond = copy.deepcopy(cond)
            new_cond = {}
            for k,v in cond.items():new_cond[k] = v

            new_cond["only_mid_control"] = self.only_mid_control
            c0 = torch.cat(new_cond["c_crossattn_1"]["image_emb"], 1)
            c1 = torch.cat(new_cond["c_crossattn_1"]["text_emb"], 1)
            fmri_vae = torch.cat(new_cond["c_crossattn_1"]["fmri_vae"],1)

            # print(x_noisy_gen.shape, t, a_t, sqrt_one_minus_at)
            # import pdb; pdb.set_trace();
            x_recon_gen, control_res, control_ctx, control_attn_qkv = self.control_model.forward_dc(x=torch.cat([x_noisy_gen], dim=1), 
                                                        timesteps=t,
                                                        c0=c0, c1=c1,
                                                        xtype='image', c0_type='vision', 
                                                        c1_type='prompt', mixed_ratio=mixed_ratio,
                                                        is_save_intermediate=is_save_intermediate,
                                                        is_save_x0=is_save_x0,
                                                        sqrt_one_minus_at=sqrt_one_minus_at,
                                                        a_t=a_t)
            # control_res = [tt.detach().requires_grad_(True) for tt in control_res]
            new_cond.pop('c_crossattn_1')
            new_cond.pop('null_prompt_emb')
            
            noisy_c_concat = control_res.pop(0)
            # fmri_control = [c * scale for c, scale in zip(control_res, self.control_scales)]
            # new_cond["control"] = fmri_control

            useful_block_idxes = [4, 5, 6]
            out_layers_injected = {}
            for useful_block_idx in reversed(useful_block_idxes):
                out_layers_injected[f"output_block_{useful_block_idx}_out_layers_features"] = control_res.pop(0)

            useful_ctx_idxes = self.useful_ctx_idxes
            injected_contexts = []
            for kk, ctx_feature in enumerate(control_ctx):
                if useful_ctx_idxes is not None and kk not in useful_ctx_idxes: continue
                injected_contexts.append(ctx_feature)
            new_cond['injected_contexts'] = injected_contexts
            new_cond['injected_attn_qkv'] = control_attn_qkv if self.mutual_selfatt else None

            ## this sentence will overwrite the obtained noisy_c_concat at inference time            
            if self.use_resnet_inject is True:
                new_cond["injected_features"] = out_layers_injected
            else:
                new_cond["injected_features"] = None

            # import pdb; pdb.set_trace()
            if self.is_no_inject is True:
                new_cond["injected_contexts"] = None
                new_cond["injected_features"] = None

            new_cond["noisy_c_concat"] = noisy_c_concat if noisy_c_concat4train is None else noisy_c_concat4train
            
            if 'layout_concat' in new_cond.keys():
                new_cond['noisy_c_concat'] = new_cond['layout_concat'][0]
                new_cond.pop('layout_concat')

            new_cond["is_return_x0"] = is_return_x0
            new_cond["sqrt_one_minus_at"] = sqrt_one_minus_at_offset
            new_cond["a_t"] = a_t_offset

            if self.is_no_concat is True:
                new_cond["noisy_c_concat"] = None

            edit_t = t_edit_in if t_edit_in is not None else t
            if is_return_x0 is False:
                x_recon_edit = self.model(x_noisy_edit, edit_t, **new_cond)
                # import pdb; pdb.set_trace();
                return x_recon_gen, x_recon_edit
            else:
                x_recon_edit, x0_recon_edit = self.model(x_noisy_edit, edit_t, **new_cond)
                # import pdb; pdb.set_trace();
                return x_recon_gen, x_recon_edit, x0_recon_edit