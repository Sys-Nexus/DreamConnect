import os.path
import glob
import json
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
import webdataset as wds


def stats_save_pickle(data_dict, path):
    with open(path, 'wb') as f:
        pickle.dump(data_dict, f)

def stats_load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def read_pkl(path, idx=0):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    coco_dict = data
    caps = []
    keys = []
    for k,v in data.items():
        caps.append(v[idx])
        keys.append(k)
    return caps, keys, coco_dict


def read_edit_json(root, map_keys):
    meta_paths = sorted(glob.glob(os.path.join(root, '*.json')))
    # print(meta_paths[:20])
    meta = []
    for meta_path in meta_paths:
        meta_i = json.load(open(meta_path, 'r'))
        meta.extend(meta_i)
    # import pdb; pdb.set_trace();
    res_dict = {}
    # for i in range(len(meta)):
    for i in range(len(map_keys)):
        res_dict[map_keys[i]] = meta[i]
    return res_dict


def load_img_from_arr(img_arr,resolution):
    image = Image.fromarray(img_arr).convert("RGB")
    w, h = resolution, resolution
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def load_img_from_string(img_path,resolution):
    # image = Image.fromarray(img_arr).convert("RGB")
    image = Image.open(img_path).convert("RGB")
    w, h = resolution, resolution
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


class NIPS23NSDDataset(Dataset):
    def __init__(self, url="nsd_data_dir/test_subj01_" + "{0..1}.tar", voxels_key='nsdgeneral.npy', split='test', resolution=320,\
            nsd_root='/data/yashengsun/Proj/MMEdit/StableDiffusionReconstruction/nsd',
            is_reconstruct_mode=False,
            reconstruct_prob=0.1,):
        super().__init__()
        self.nsda = NSDAccess(nsd_root)
        self.is_reconstruct_mode = is_reconstruct_mode
        self.reconstruct_prob = reconstruct_prob

        cached_path = 'datadict_{}_{}.pkl'.format(split, 'subj01')
        self.resolution = resolution
        # import pdb; pdb.set_trace()
        if os.path.exists(cached_path):
            with open(cached_path, 'rb') as f:
                self.data_dict = pickle.load(f)
            self.cocos, self.voxels = self.data_dict['cocos'], self.data_dict['voxels']
        else:
            self.voxels, self.cocos = [], []
            dl = wds.WebDataset(url, resampled=False)\
                    .decode("torch")\
                    .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
                    .to_tuple("voxels", "images", "coco")\
                    .batched(1, partial=False)
            
            for idx, (voxel, img, coco) in enumerate(tqdm(dl)):
                if split == 'test':
                    # self.voxels.append(torch.mean(voxel,axis=1))
                    self.voxels.append(np.mean(voxel,axis=1))
                if split == 'train':
                    self.voxels.append(voxel)
                self.cocos.append(coco.item())
                self.data_dict = {'voxels':self.voxels, 'cocos':self.cocos}
            with open(cached_path, 'wb') as f:
                pickle.dump(self.data_dict, f)
        
        # print(self.cocos)
        nsd_root = os.path.dirname(os.path.abspath(__file__))
        nsd_coco_caption_path = os.path.join(nsd_root, 'misc/nsd_coco_caption.pkl')
        self.caps, self.keys, self.cap_dict = read_pkl(nsd_coco_caption_path)
        self.edited_root = '/data/yashengsun/Proj/Diffusion/InstructDiffusion/nsd_coco_output'
        self.meta_info = read_edit_json(os.path.join(nsd_root, 'misc'), self.keys)
        self.valid_do_nothing_ops = [' ']

    def __getitem__(self, index):
        s = self.cocos[index]
        # voxel, img_input, coco = self.data[index]
        caps = self.cap_dict[s]
        img = self.nsda.read_images(s)
        init_image = load_img_from_arr(img, self.resolution)
        init_image = repeat(init_image, '1 ... -> b ...', b=1)
        fmri_norm = self.voxels[index][0]

        nsd_dict = {'cap': random.choices(caps)[0], 'image': init_image[0], 'fmri': fmri_norm}

        if self.is_reconstruct_mode or random.uniform(0,1.)<self.reconstruct_prob:
            instruction_text = random.choice(self.valid_do_nothing_ops)
            nsd_dict['fmri_edit'] = {'c_concat': init_image[0], 'c_crossattn': instruction_text, 'c_crossattn_1': fmri_norm}
            nsd_dict['edited'] = init_image[0]
        else:
            try:
                chosen_i = 0
                instruction_text = self.meta_info[s]['edit'][chosen_i]
                nsd_dict['fmri_edit'] = {'c_concat': init_image[0], 'c_crossattn': instruction_text, 'c_crossattn_1': fmri_norm}
                edited_path = os.path.join(self.edited_root, '{:06d}'.format(s), 'output_{:06d}_seed93151_id{}.jpg'.format(s,chosen_i))
                nsd_dict['edited'] = load_img_from_string(edited_path, self.resolution)[0] # TODO
            except:
                # import tracebrack
                # tracebrack.print_exc()
                ## If the triplet pairs do not exist, use do nothing operation
                instruction_text = random.choice(self.valid_do_nothing_ops)
                nsd_dict['fmri_edit'] = {'c_concat': init_image[0], 'c_crossattn': instruction_text, 'c_crossattn_1': fmri_norm}
                nsd_dict['edited'] = init_image[0]
        # print(nsd_dict['cap'], nsd_dict['fmri_edit']['c_crossattn'])
        return nsd_dict

    def __len__(self):
        return len(self.cocos)

class NSDDataset(Dataset):
    ## it seems that ventral area is sensitive to captions
    def __init__(self, nsd_root, idxes, batch_size=1, resolution=320, split='train', subject='subj01', 
                is_reconstruct_mode=False,
                reconstruct_prob=0.1,
                # roi=['early', 'ventral', 'midventral', 'midlateral', 'lateral', 'parietal'], 
                roi=['early'], 
                target='context'):
        self.nsda = NSDAccess(nsd_root)
        self.idxes = idxes
        self.reconstruct_prob=reconstruct_prob
        self.batch_size = batch_size
        self.resolution = resolution
        self.split = split
        self.is_reconstruct_mode = is_reconstruct_mode

        mridir = f'{os.path.dirname(nsd_root)}/mrifeat/{subject}/'

        X = []
        X_te = []
        for croi in roi:
            if 'conv' in target: # We use averaged features for GAN due to large number of dimension of features
                cX = np.load(f'{mridir}/{subject}_{croi}_betas_ave_tr.npy').astype("float32")
            else:
                cX = np.load(f'{mridir}/{subject}_{croi}_betas_tr.npy').astype("float32")
            cX_te = np.load(f'{mridir}/{subject}_{croi}_betas_ave_te.npy').astype("float32")
            # print(split, target)
            # import pdb; pdb.set_trace();
            X.append(cX)
            X_te.append(cX_te)
        X = np.hstack(X)
        X_te = np.hstack(X_te)
        self.X = X
        self.X_te = X_te

        nsd_root = os.path.dirname(os.path.abspath(__file__))
        stats_path = os.path.join(nsd_root, 'misc/stats_{}.pkl'.format('_'.join(roi)+'_'+target))
        if split == 'train' and not os.path.exists(stats_path):
            self.X_mean, self.X_std = X.mean(axis=0,keepdims=True), X.std(axis=0,keepdims=True)
            stats = {'X_mean': self.X_mean, 'X_std': self.X_std}
            stats_save_pickle(stats, stats_path)
        else:
            stats = stats_load_pickle(stats_path)
            self.X_mean, self.X_std = stats['X_mean'], stats['X_std']
        # print('X shape: ', X.shape)
        # import pdb; pdb.set_trace()

        # vae_root = '/data/yashengsun/Proj/MMEdit/StableDiffusionReconstruction/nsdfeat_256/init_latent'
        # self.vae_paths = {s: os.path.join(vae_root, '{:06d}.npy'.format(s)) for s in self.idxes}

        nsd_coco_caption_path = os.path.join(nsd_root, 'misc/nsd_coco_caption.pkl')
        self.caps, self.keys, self.cap_dict = read_pkl(nsd_coco_caption_path)
        # import pdb; pdb.set_trace();

        self.meta_info = read_edit_json(os.path.join(nsd_root, 'misc'), self.keys)
        self.edited_root = '/data/yashengsun/Proj/Diffusion/InstructDiffusion/nsd_coco_output'

        self.valid_do_nothing_ops = [' ']

    def __getitem__(self, index):
        s = self.idxes[index]
        # m_idx = self.mri_idxes[index]
        # prompt = []
        # prompts = self.nsda.read_image_coco_info([s], info_type='captions')
        # for p in prompts:
        #     prompt.append(p['caption'])

        caps = self.cap_dict[s]

        img = self.nsda.read_images(s)
        init_image = load_img_from_arr(img, self.resolution)
        init_image = repeat(init_image, '1 ... -> b ...', b=self.batch_size)

        fmri_norm = (self.X[index]-self.X_mean)/self.X_std
        ## TODO: use random.choices()
        # nsd_dict = {'cap': caps[0], 'prompt': prompt[0],  'image': init_image[0], 'fmri': fmri_norm}
        nsd_dict = {'cap': random.choices(caps)[0], 'image': init_image[0], 'fmri': fmri_norm, 's': s}

        # image_vae = np.load(self.vae_paths[s])
        # nsd_dict['image_vae'] = image_vae

        # TODO: currently , only use the first edit instruction

        if self.is_reconstruct_mode or random.uniform(0,1.)<self.reconstruct_prob:
            instruction_text = random.choice(self.valid_do_nothing_ops)
            nsd_dict['fmri_edit'] = {'c_concat': init_image[0], 'c_crossattn': instruction_text, 'c_crossattn_1': fmri_norm}
            # nsd_dict['fmri_edit'] = {'c_concat': init_image[0], 'c_crossattn': random.choices(caps)[0], 'c_crossattn_1': fmri_norm}
            nsd_dict['edited'] = init_image[0]
        else:
            try:
                chosen_i = 0
                instruction_text = self.meta_info[s]['edit'][chosen_i]
                nsd_dict['fmri_edit'] = {'c_concat': init_image[0], 'c_crossattn': instruction_text, 'c_crossattn_1': fmri_norm}
                edited_path = os.path.join(self.edited_root, '{:06d}'.format(s), 'output_{:06d}_seed93151_id{}.jpg'.format(s,chosen_i))
                nsd_dict['edited'] = load_img_from_string(edited_path, self.resolution)[0] # TODO
            except:
                ## If the triplet pairs do not exist, use do nothing operation
                instruction_text = random.choice(self.valid_do_nothing_ops)
                nsd_dict['fmri_edit'] = {'c_concat': init_image[0], 'c_crossattn': instruction_text, 'c_crossattn_1': fmri_norm}
                nsd_dict['edited'] = init_image[0]
        return nsd_dict

    def __len__(self):
        if self.split == 'train': return self.X.shape[0]
        else: return self.X_te.shape[0]



if __name__ == '__main__':
    dataset = NIPS23NSDDataset()
    for i, in_dict in enumerate(dataset):
        # print(in_dict.keys())
        # print(in_dict[''])
        if i>20: import pdb; pdb.set_trace();