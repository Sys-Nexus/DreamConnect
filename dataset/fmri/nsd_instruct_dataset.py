import os
import scipy.io

import sys
from third_party.StableDiffusionReconstruction.codes.utils.nsd_creater import create_nsd_dataset, NSDDataset
import os.path

from tqdm import tqdm
import random
import numpy as np
import scipy.io
from PIL import Image
import torch
import pickle
import PIL
import sys
import cv2

import torchvision.transforms as transforms
from einops import rearrange
import copy
import torch
import numpy as np
import glob


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


## here we write a wrapper for NSDDataset, simply move create_nsd_dataset here
class NSDInstructDataset(NSDDataset):
    def __init__(self, nsd_root, resolution=320, use_stim='each', subject='subj01', split='train', batch_size=1):
        # super().__init__()
        nsd_expdesign = scipy.io.loadmat(os.path.join(nsd_root, 'nsddata/experiments/nsd/nsd_expdesign.mat'))
        # Note that most of them are 1-base index!
        # This is why I subtract 1
        sharedix = nsd_expdesign['sharedix'] - 1

        if use_stim == 'ave':
            stims = np.load(f'{os.path.dirname(nsd_root)}/mrifeat/{subject}/{subject}_stims_ave.npy')
        else:  # Each
            stims = np.load(f'{os.path.dirname(nsd_root)}/mrifeat/{subject}/{subject}_stims.npy')
        # print('stims shape: ', stims.shape)
        train_idxes, test_idxes = [], []
        mri_train_idxes, mri_test_idxes = [], []
        for idx, s in tqdm(enumerate(stims)):
            if s in sharedix:
                test_idxes.append(s)
                mri_test_idxes.append(idx)
            else:
                train_idxes.append(s)
                mri_train_idxes.append(idx)

        idxes = train_idxes if split == 'train' else test_idxes
        super().__init__(nsd_root, idxes, batch_size=batch_size, resolution=resolution, split=split, subject=subject)
