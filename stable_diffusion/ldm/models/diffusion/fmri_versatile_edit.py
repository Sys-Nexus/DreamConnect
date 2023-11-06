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


class VersatileControlNet(UNetModelVD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dims = 2
        self.zero_convs = nn.ModuleList([self.make_zero_conv(self.model_channels)])
        for level_idx, mult in enumerate(self.channel_mult):
            for _ in range(self.num_noattn_blocks[level_idx]):
                ch = mult * self.model_channels
                self.zero_convs.append(self.make_zero_conv(ch))
            if level_idx != len(self.channel_mult) - 1:
                self.zero_convs.append(self.make_zero_conv(ch))
        self.middle_block_out = self.make_zero_conv(ch)

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward_dc(self, x, timesteps, c0, c1, xtype, c0_type, c1_type, mixed_ratio):
        outs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        
        x=x.half()
        emb = self.time_embed(t_emb.half())

        if xtype == 'text':
            x = x[:, :, None, None]
        h = x
        for i_module, t_module, zero_conv in zip(self.unet_image.input_blocks, self.unet_text.input_blocks, self.zero_convs):
            h = self.mixed_run_dc(i_module, t_module, h, emb, c0, c1, xtype, c0_type, c1_type, mixed_ratio)
            outs.append(zero_conv(h, emb))

        h = self.mixed_run_dc(self.unet_image.middle_block, self.unet_text.middle_block, 
                                h, emb, c0, c1, xtype, c0_type, c1_type, mixed_ratio)
        outs.append(self.middle_block_out(h, emb))
        return outs


def freeze_params(model):
    # model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

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



class ControlLDM(LatentDiffusion):

    def __init__(self, clip_cfg, *args, **kwargs):
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
            # import pdb; pdb.set_trace();

    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None, uncond=0.075, sz=256):
        # x = super().get_input(batch, k)
        x = DDPM.get_input(self, batch, k)
        if bs is not None:
            x = x[:bs]
        if sz is not None:
            x = F.interpolate(x, (sz,sz))
        # x = x.half()
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        cond_key = cond_key or self.cond_stage_key

        xc = DDPM.get_input(self, batch, cond_key)
        cap = batch['cap']
        if bs is not None:
            xc["c_crossattn"] = xc["c_crossattn"][:bs]
            xc["c_crossattn_1"] = xc["c_crossattn_1"][:bs]
            xc["c_concat"] = xc["c_concat"][:bs]
            cap = batch['cap'][:bs]
        if sz is not None:
            xc["c_concat"] = F.interpolate(xc["c_concat"], (sz,sz))
        x, xc["c_concat"], xc["c_crossattn_1"] = x.to(z), xc["c_concat"].to(z), xc["c_crossattn_1"].to(z)
        # x, xc["c_concat"], xc["c_crossattn_1"] = x.half(), xc["c_concat"].half(), xc["c_crossattn_1"].half()
        # import pdb; pdb.set_trace();
        cond = {}
        random = torch.rand(x.size(0), device=z.device)
        prompt_mask = rearrange(random < uncond, "n -> n 1 1")
        fmri_prompt_mask = (random >= uncond).float() * (random < uncond*2).float()
        fmri_prompt_mask = rearrange(fmri_prompt_mask, "n -> n 1 1")
        input_mask = 1 - rearrange((random >= uncond*2).float() * (random < uncond*3).float(), "n -> n 1 1 1")
        
        null_prompt = self.get_learned_conditioning([""])
        # fmri_null_prompt = self.get_learned_conditioning_fmri(torch.zeros_like(xc["c_crossattn_1"]))
        # fmri_learned_prompt = self.get_learned_conditioning_fmri(xc["c_crossattn_1"])
        # cond["c_crossattn_1"] = [torch.where(fmri_prompt_mask.bool(), fmri_null_prompt, fmri_learned_prompt)]

        null_x = torch.zeros_like(x)
        null_cap = ['' for _ in range(len(cap))]
        fmri_null_x = self.vd_clip.clip_encode_vision(null_x)
        fmri_null_cap = self.vd_clip.clip_encode_text(null_cap)

        fmri_x = self.vd_clip.clip_encode_vision(x)
        fmri_cap = self.vd_clip.clip_encode_text(cap)

        # import pdb;pdb.set_trace()
        cond["c_crossattn_1"] = {}
        cond["c_crossattn_1"]["image_emb"] = [torch.where(fmri_prompt_mask.bool(), fmri_null_x, fmri_x)]
        cond["c_crossattn_1"]["text_emb"] = [torch.where(fmri_prompt_mask.bool(), fmri_null_cap, fmri_cap)]

        cond["c_crossattn"] = [torch.where(prompt_mask, null_prompt, self.get_learned_conditioning(xc["c_crossattn"]).detach())]

        if self.is_fmri_input is True:
            cond["c_concat"] = [input_mask * self.fmri2visual_model((xc["c_concat"])).detach()]
        else:
            cond["c_concat"] = [input_mask * self.encode_first_stage((xc["c_concat"])).mode().detach()]

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
                   N=2, n_row=4, sample=True, 
                   steps=100, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=False):
        # import pdb; pdb.set_trace()
        N = min(batch['image'].shape[0], N)
        s = batch['s'][0]

        self.model.eval()
        use_ddim = False

        log = dict()
        z_gt, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N, uncond=0)
        cap = batch['cap'][:N]

        cond = {}
        cond["c_crossattn"] = [self.get_learned_conditioning(xc["c_crossattn"])]
        # cond["c_crossattn_1"] = [self.get_learned_conditioning_fmri(xc["c_crossattn_1"])]
        # import pdb; pdb.set_trace();
        cond["c_crossattn_1"] = {}
        cond["c_crossattn_1"]["image_emb"] = self.vd_clip.clip_encode_vision(x)
        cond["c_crossattn_1"]["text_emb"] = self.vd_clip.clip_encode_text(cap)

        uncond = {}
        null_prompt = self.get_learned_conditioning([""]*N)
        # fmri_null_prompt = self.get_learned_conditioning_fmri(torch.zeros_like(xc["c_crossattn_1"]))
        uncond["c_crossattn"] = [null_prompt]
        # uncond["c_crossattn_1"] = [fmri_null_prompt]
        uncond["c_crossattn_1"] = {}
        uncond["c_crossattn_1"]["image_emb"] = self.vd_clip.clip_encode_vision(torch.zeros_like(x))
        uncond["c_crossattn_1"]["text_emb"] = self.vd_clip.clip_encode_text(['' for i in range(len(cap))])

        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": cfg_text,
            "fmri_cfg_scale": cfg_fmri,
        }

        # import pdb; pdb.set_trace()

        sigmas = model_wrap.get_sigmas(steps)
        z_pred = torch.randn_like(z_gt) * sigmas[0]
        z_pred = K.sampling.sample_euler_ancestral(model_wrap_cfg, z_pred, sigmas, extra_args=extra_args)
        x_pred = self.decode_first_stage(z_pred)

        if True:
            import pdb; pdb.set_trace();
            sigmas = model_wrap.get_sigmas(steps)
            new_steps = int(0.75*steps)
            sigmas_clamp = sigmas[-new_steps:]
            noisy_steps = torch.ones(size=(c['c_concat'][0].shape[0],)).to(sigmas.device).long()*int(0.25*steps)
            z_concat_noised = self.q_sample(c['c_concat'][0], noisy_steps)
            z_pred_w_spatial = z_concat_noised * sigmas_clamp[0]
            z_pred_w_spatial = K.sampling.sample_euler_ancestral(model_wrap_cfg, z_pred_w_spatial, sigmas_clamp, extra_args=extra_args)
            x_pred_w_spatial = self.decode_first_stage(z_pred_w_spatial)
            torchvision.utils.save_image(torch.cat([x_pred, x_pred_w_spatial],dim=2), 'output_concat.jpg')
            import pdb; pdb.set_trace();

        # import pdb; pdb.set_trace();
        log["gt"] = x
        log["concat"] = xc["c_concat"]
        log["recon"] = xrec
        log["samples"] = x_pred
        log["instruction"] = log_txt_as_img((x.shape[2], x.shape[3]), xc["c_crossattn"])

        for k in log.keys():
            root = os.path.join(save_dir, "images", split)
            filename = "{}_iter-{:06}_ep-{:06}_bidx-{:06d}-{:06d}.png".format(k, iter_n, epoch_n, batch_idx, s)
            path = os.path.join(root, filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torchvision.utils.save_image(log[k]*0.5+0.5, path)

        cats = [log['gt'].detach().cpu(), log['instruction'].detach().cpu(), log['concat'].detach().cpu(), log['samples'].detach().cpu()]
        cats = torch.concat(cats, dim=-2)
        filename = "all_iter-{:06}_ep-{:06}_bidx-{:06d}-{:06d}.png".format(iter_n, epoch_n, batch_idx, s)
        path = os.path.join(root, filename)
        torchvision.utils.save_image(cats*0.5+0.5, path)

        self.model.train()

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        
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
            cond["only_mid_control"] = self.only_mid_control
            # control_prompt = torch.cat(cond["c_crossattn_1"] , 1)
            ## TODO: hard code to set the hint to zero
            # fmri_control = self.control_model(x=torch.cat([x_noisy], dim=1), 
            #                                     timesteps=t, context=control_prompt)

            # c0 = self.vd_clip.clip_encode_vision(cond['c_crossattn_1']['image'])
            # c1 = self.vd_clip.clip_encode_text(cond['c_crossattn_1']['text'])
            c0 = torch.cat(cond["c_crossattn_1"]["image_emb"], 1)
            c1 = torch.cat(cond["c_crossattn_1"]["text_emb"], 1)
            # import pdb; pdb.set_trace();

            control_res = self.control_model.forward_dc(x=torch.cat([x_noisy], dim=1), timesteps=t,
                                                        c0=c0, c1=c1,
                                                        xtype='image', c0_type='vision', 
                                                        c1_type='prompt', mixed_ratio=0.6)
            cond.pop('c_crossattn_1')

            fmri_control = [c * scale for c, scale in zip(control_res, self.control_scales)]
            cond["control"] = fmri_control
            ## only add above 
            x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

######## for testing the versatile diffusion
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

        if True:
            pretrained_control_unet_path = 'third_party/versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
            pretrained_state_dict = torch.load(pretrained_control_unet_path, map_location="cpu")
            # import pdb; pdb.set_trace();
            pretrained_state_dict = {k.replace('autokl.',''):v for k,v in pretrained_state_dict.items() if 'autokl.' in k}
            missing, unexpected = net.autokl.load_state_dict(pretrained_state_dict, strict=False)

            print('autokl missing {} params.'.format(len(missing)))
            print('autokl missing: ', missing)
            print('autokl unexpected {} params.'.format(len(unexpected)))

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

        sampler.model.model.diffusion_model.half()
        
        self.sampler = sampler
        self.net = net
        freeze_params(self.net)
        self.net.eval()

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
        zim = batch['image']
        # zin = zim*2 - 1
        zin = zim.half().cuda()

        self.sampler.model.model.diffusion_model.device = zin.device
        self.net.device = zin.device

        init_latent = self.net.autokl_encode(zin)
        self.sampler.make_schedule(ddim_num_steps=self.ddim_steps, ddim_eta=self.ddim_eta, verbose=False)

        dummy = ''
        utx = self.net.clip_encode_text(dummy)
        utx = utx.half()

        # import pdb; pdb.set_trace();
        dummy = torch.zeros((1,3,224,224)).half().cuda()
        uim = self.net.clip_encode_vision(dummy)
        uim = uim.half()
        
        cim = self.net.clip_encode_vision(zin)
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
        
        x = self.net.autokl_decode(z.half())
        x = torch.clamp((x+1.0)/2.0, min=0.0, max=1.0)
        import torchvision
        torchvision.utils.save_image(x, 'x.jpg')
        torchvision.utils.save_image((1.+batch['image'])*0.5, 'gt.jpg')
        import pdb; pdb.set_trace();