from types import SimpleNamespace
import sys
import os
import yaml
from easydict import EasyDict
import argparse
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import argparse
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import kornia
from kornia.augmentation.container import AugmentationSequential
img_augment = AugmentationSequential(
    kornia.augmentation.RandomResizedCrop((224,224), (0.9,1), p=0.3),
    kornia.augmentation.Resize((224, 224)),
)

from einops import rearrange
import einops
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
import k_diffusion as K
import torchvision
import torchvision.transforms as transforms

import sys
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_root)
sys.path.append(os.path.join(proj_root,"stable_diffusion"))

from ldm.util import instantiate_from_config



class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model


def init_gen_model(ckpt_path = 'stable_diffusion/models/ldm/stable-diffusion-v1/MagicBrush-epoch-52-step-4999.ckpt'):
    config = OmegaConf.load('configs/test/generate.yaml')
    model = load_model_from_config(config, ckpt_path, vae_ckpt=None)

    model.eval().cuda()
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])
    steps = 100
    sigmas = model_wrap.get_sigmas(steps)

    return model, model_wrap, model_wrap_cfg, sigmas, null_token

                                                
def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--method', type=str, default='magic_brush')
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--recon_root', type=str, default='logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val/')
    # parser.add_argument('--instruct_root', type=str, default='logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val/')
    args = parser.parse_args()

    dataset_cfg_str = """
    train:
      target: third_party.StableDiffusionReconstruction.codes.utils.nsd_creater.NIPS23NSDDataset
      params:
        nsd_root: '/data/yashengsun/Proj/MMEdit/StableDiffusionReconstruction/nsd'
        resolution: 320
        split: 'train'
        is_reconstruct_mode: False
        url: 'nsd_data_dir/train_subj01_{0..17}.tar'
        reconstruct_prob: 0.0

    validation:
      target: third_party.StableDiffusionReconstruction.codes.utils.nsd_creater.NIPS23NSDDataset
      params:
        nsd_root: '/data/yashengsun/Proj/MMEdit/StableDiffusionReconstruction/nsd'
        resolution: 320
        split: 'test'
        is_reconstruct_mode: False
        url: 'nsd_data_dir/test_subj01_{0..1}.tar'
        reconstruct_prob: 0.0
        use_first_instruct: True
    """
    dataset_cfg = EasyDict(yaml.safe_load(dataset_cfg_str))
    val_dataset = instantiate_from_config(dataset_cfg['validation'])

    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=False)

    if args.method == 'magic_brush':
        args.ckpt_path = 'stable_diffusion/models/ldm/stable-diffusion-v1/MagicBrush-epoch-52-step-4999.ckpt'
    elif args.method == 'inst_dif':
        args.ckpt_path = 'stable_diffusion/models/ldm/stable-diffusion-v1/v1-5-pruned-emaonly-adaption-task-humanalign.ckpt'
    elif args.method == 'inst_pix2pix':
        args.ckpt_path = 'stable_diffusion/models/ldm/stable-diffusion-v1/instruct-pix2pix-00-22000.ckpt'
    else:
        raise ValueError

    args.instruct_root = args.recon_root.replace('images/val', 'images/val_{}'.format(args.method))
    os.makedirs(args.instruct_root, exist_ok=True)

    model, model_wrap, model_wrap_cfg, sigmas, null_token = init_gen_model(ckpt_path=args.ckpt_path)
    image_path_templ = 'all_iter-999999_ep-999999_bidx-{:06d}-{:06d}.png'
    for val_i, batch_dict in tqdm(enumerate(val_dl)):
        # import pdb; pdb.set_trace()
        image_name = image_path_templ.format(val_i, batch_dict['s'].item())
        image_path = os.path.join(args.recon_root, image_name)
        cond = {}
        instruct_text = batch_dict['fmri_edit']['c_crossattn']
        cond["c_crossattn"] = [model.get_learned_conditioning(instruct_text)]
        input_image_all = transforms.ToTensor()(Image.open(image_path))
        input_image = input_image_all[:,3*512:4*512,:].unsqueeze(0).to(device='cuda')
        # input_image = F.interpolate(input_image, size=(256,224))
        cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

        uncond = {}
        uncond["c_crossattn"] = [null_token]
        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": 7.5,
            "image_cfg_scale": 1.5,
        }

        z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        x = model.decode_first_stage(z)
        input_image_all[:,4*512:,:] = x[0,]
        save_name = image_name.replace(args.recon_root, args.instruct_root)
        # import pdb; pdb.set_trace();
        torchvision.utils.save_image(input_image_all, os.path.join(args.instruct_root, save_name))


if __name__ == '__main__':
    main()
