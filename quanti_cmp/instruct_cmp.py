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

import sys
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_root)
sys.path.append(os.path.join(proj_root,"stable_diffusion"))

from ldm.util import instantiate_from_config


def init_gen_model(ckpt_path = 'stable_diffusion/models/ldm/stable-diffusion-v1/MagicBrush-epoch-52-step-4999.ckpt'):
    config = OmegaConf.load('configs/generate.yaml')
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
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
