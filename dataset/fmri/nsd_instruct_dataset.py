import os

import sys
from third_party.StableDiffusionReconstruction.codes.utils.nsd_creater import create_nsd_dataset

import torchvision.transforms as transforms
from einops import rearrange
import copy
import torch
import numpy as np


def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img

class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img

def fmri_transform(x, sparse_rate=0.2):
    # x: 1, num_voxels
    x_aug = copy.deepcopy(x)
    idx = np.random.choice(x.shape[0], int(x.shape[0]*sparse_rate), replace=False)
    x_aug[idx] = 0
    return torch.FloatTensor(x_aug)


def channel_last(img):
    if img.shape[-1] == 3:
        return img
    return rearrange(img, 'c h w -> h w c')


@registry.register_builder("nsd")
class NSDBuilder(BaseDatasetBuilder):
    # train_dataset_cls = LaionDataset

    # DATASET_CONFIG_DICT = {"default": "configs/datasets/laion/defaults_2B_multi.yaml"}
    # DATASET_CONFIG_DICT = {"default": "configs/datasets/fmri/bold5000.yaml"}
    DATASET_CONFIG_DICT = {"default": "configs/datasets/fmri/nsd.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        # import pdb; pdb.set_trace()
        datasets = dict()

        img_transform_train = transforms.Compose([
            normalize,
            # random_crop(config.img_size - crop_pix, p=0.5),
            transforms.Resize((224, 224)),
            # channel_last
        ])
        img_transform_test = transforms.Compose([
            normalize, transforms.Resize((224, 224)),
            # channel_last
        ])

        ## here we hard-coded the condition
        nsd_root = '/data/yashengsun/Proj/MMEdit/StableDiffusionReconstruction/nsd'
        fmri_latents_dataset_train, fmri_latents_dataset_test = \
            create_nsd_dataset(nsd_root)
        datasets['train'] = fmri_latents_dataset_train
        datasets['test'] = fmri_latents_dataset_test
        # import pdb; pdb.set_trace()
        return datasets
