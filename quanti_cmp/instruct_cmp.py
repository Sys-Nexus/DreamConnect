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
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
import k_diffusion as K
import torchvision

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
    args = parser.parse_args()

    dataset_cfg_str = """
    train:
      target: third_party.StableDiffusionReconstruction.codes.utils.nsd_creater.NIPS23NSDDataset
      params:
        nsd_root: '/data/yashengsun/Proj/MMEdit/StableDiffusionReconstruction/nsd'
        resolution: 320
        split: 'train'
        is_reconstruct_mode: True
        url: 'nsd_data_dir/train_subj01_{0..17}.tar'
        reconstruct_prob: 1.05

    validation:
      target: third_party.StableDiffusionReconstruction.codes.utils.nsd_creater.NIPS23NSDDataset
      params:
        nsd_root: '/data/yashengsun/Proj/MMEdit/StableDiffusionReconstruction/nsd'
        resolution: 320
        split: 'test'
        is_reconstruct_mode: True
        url: 'nsd_data_dir/test_subj01_{0..1}.tar'
        reconstruct_prob: 1.05
    """
    dataset_cfg = EasyDict(yaml.safe_load(dataset_cfg_str))
    val_dataset = instantiate_from_config(dataset_cfg['validation'])

    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=False)

    if args.method == 'magic_brush':
        args.ckpt_path = 'stable_diffusion/models/ldm/stable-diffusion-v1/MagicBrush-epoch-52-step-4999.ckpt'
    else:
        raise ValueError

    model, model_wrap, model_wrap_cfg, sigmas, null_token = init_gen_model(ckpt_path=args.ckpt_path)
    image_path_templ = 'all_iter-999999_ep-999999_bidx-{:06d}-{:06d}.png'
    for val_i, batch_dict in tqdm(enumerate(val_dl)):
        import pdb; pdb.set_trace()
        # image_path = image_path_templ.format(val_i, )

if __name__ == '__main__':
    main()
