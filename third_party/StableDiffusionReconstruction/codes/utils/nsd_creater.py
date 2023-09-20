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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nsd_access import NSDAccess
from einops import repeat
from torch.utils.data import Dataset


def load_img_from_arr(img_arr,resolution):
    image = Image.fromarray(img_arr).convert("RGB")
    w, h = resolution, resolution
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def create_nsd_dataset(nsd_root, batch_size=1, resolution=320, use_stim='each', subject='subj01'):

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

    nsd_dataset_train = NSDDataset(nsd_root, train_idxes, batch_size=batch_size, resolution=resolution, split='train', subject=subject)
    nsd_dataset_test = NSDDataset(nsd_root, test_idxes, batch_size=batch_size, resolution=resolution, split='test', subject=subject)

    # import pdb; pdb.set_trace()
    return nsd_dataset_train, nsd_dataset_test


class NSDDataset(Dataset):
    ## it seems that ventral area is sensitive to captions
    def __init__(self, nsd_root, idxes, batch_size=1, resolution=320, split='train', subject='subj01', roi=['ventral'], target='c'):
        self.nsda = NSDAccess(nsd_root)
        self.idxes = idxes
        # self.mri_idxes = mri_idxes
        self.batch_size = batch_size
        self.resolution = resolution
        self.split = split

        mridir = f'{os.path.dirname(nsd_root)}/mrifeat/{subject}/'

        X = []
        X_te = []
        for croi in roi:
            if 'conv' in target: # We use averaged features for GAN due to large number of dimension of features
                cX = np.load(f'{mridir}/{subject}_{croi}_betas_ave_tr.npy').astype("float32")
            else:
                cX = np.load(f'{mridir}/{subject}_{croi}_betas_tr.npy').astype("float32")
            cX_te = np.load(f'{mridir}/{subject}_{croi}_betas_ave_te.npy').astype("float32")
            X.append(cX)
            X_te.append(cX_te)
        X = np.hstack(X)
        X_te = np.hstack(X_te)
        self.X = X
        self.X_te = X_te
        self.X_mean, self.X_std = X.mean(axis=0,keepdims=True), X.std(axis=0,keepdims=True)
        # print('X shape: ', X.shape)
        # import pdb; pdb.set_trace()

    def __getitem__(self, index):
        s = self.idxes[index]
        # m_idx = self.mri_idxes[index]
        prompt = []
        prompts = self.nsda.read_image_coco_info([s], info_type='captions')
        for p in prompts:
            prompt.append(p['caption'])
        img = self.nsda.read_images(s)

        init_image = load_img_from_arr(img, self.resolution)
        init_image = repeat(init_image, '1 ... -> b ...', b=self.batch_size)
        nsd_dict = {'cap': random.choice(prompt), 'image': init_image, 'fmri': (self.X[index]-self.X_mean)/self.X_std}
        return nsd_dict

    def __len__(self):
        if self.split == 'train': return self.X.shape[0]
        else: return self.X_te.shape[0]
        # return 100
        # return 1