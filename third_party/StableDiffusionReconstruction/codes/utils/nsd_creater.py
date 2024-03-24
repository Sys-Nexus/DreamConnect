import os.path
import glob
import json
from tqdm import tqdm
import random
import numpy as np
# import scipy.io
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
    # import pdb; pdb.set_trace();
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
            image_clip_root='/data/yashengsun/Proj/MMEdit/fMRIInstructDiffusion/img_clip',
            is_reconstruct_mode=False,
            reconstruct_prob=0.1,):
        super().__init__()
        self.nsda = NSDAccess(nsd_root)
        self.is_reconstruct_mode = is_reconstruct_mode
        self.reconstruct_prob = reconstruct_prob
        self.image_clip_root = image_clip_root

        sub = 1
        self.nsd_cliptext_path = 'nsd_data_dir/predicted_features/subj{:02d}/nsd_cliptext_pred{}_nsdgeneral.npy'.format(sub,split)
        self.nsd_clipvision_path = 'nsd_data_dir/predicted_features/subj{:02d}/nsd_clipvision_pred{}_nsdgeneral.npy'.format(sub,split)
        if os.path.exists(self.nsd_cliptext_path):
            self.all_nsd_cliptext = np.load(self.nsd_cliptext_path)
            self.all_nsd_clipvision = np.load(self.nsd_clipvision_path)

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
        # import pdb; pdb.set_trace()
        self.edited_root = '/data/yashengsun/Proj/Diffusion/InstructDiffusion/nsd_coco_output'
        self.meta_info = read_edit_json(os.path.join(nsd_root, 'misc'), self.keys)
        self.valid_do_nothing_ops = [' ']

    def __getitem__(self, index):
        s = self.cocos[index]
        image_clip_path = os.path.join(self.image_clip_root, '{:05d}.npy'.format(s))
        # voxel, img_input, coco = self.data[index]
        caps = self.cap_dict[s]
        img = self.nsda.read_images(s)
        init_image = load_img_from_arr(img, self.resolution)
        init_image = repeat(init_image, '1 ... -> b ...', b=1)
        fmri_norm = self.voxels[index][0]

        nsd_dict = {'cap': random.choices(caps)[0], 'image': init_image[0], 
                    'fmri': fmri_norm, 's': s}
        
        if os.path.exists(image_clip_path):
            nsd_dict['img_clip'] = np.load(image_clip_path)
        if os.path.exists(self.nsd_cliptext_path):
            nsd_cliptext = self.all_nsd_cliptext[index]
            nsd_clipvision = self.all_nsd_clipvision[index]
            nsd_dict['nsd_cliptext'] = nsd_cliptext
            nsd_dict['nsd_clipvision'] = nsd_clipvision

        if self.is_reconstruct_mode or random.uniform(0,1.)<self.reconstruct_prob:
            instruction_text = random.choice(self.valid_do_nothing_ops)
            nsd_dict['fmri_edit'] = {'c_concat': init_image[0], 'c_crossattn': instruction_text, 'c_crossattn_1': fmri_norm}
            nsd_dict['edited'] = init_image[0]
        else:
            # try:
            chosen_pool = []
            for chosen_i in [0,1]:
                edited_path = os.path.join(self.edited_root, '{:06d}'.format(s), 'output_{:06d}_seed93151_id{}.jpg'.format(s,chosen_i))
                if os.path.exists(edited_path):
                    chosen_pool.append(chosen_i)

            if len(chosen_pool) and s in self.meta_info and 'edit' in self.meta_info[s] and max(chosen_pool) < len(self.meta_info[s]['edit']):
                chosen_i = random.choice(chosen_pool)
                instruction_text = self.meta_info[s]['edit'][chosen_i]
                output_text = self.meta_info[s]['output'][chosen_i]
                nsd_dict['fmri_edit'] = {'c_concat': init_image[0], 'output': output_text, 'c_crossattn': instruction_text, 'c_crossattn_1': fmri_norm}
                edited_path = os.path.join(self.edited_root, '{:06d}'.format(s), 'output_{:06d}_seed93151_id{}.jpg'.format(s,chosen_i))
                nsd_dict['edited'] = load_img_from_string(edited_path, self.resolution)[0] # TODO
            # except:
                # import traceback
                # traceback.print_exc()
                # print('chosen : ', chosen_i, 's: ', s, 'cap: ', caps, 'instru: ', instruction_text)
                ## If the triplet pairs do not exist, use do nothing operation
            else:
                instruction_text = random.choice(self.valid_do_nothing_ops)
                nsd_dict['fmri_edit'] = {'c_concat': init_image[0], 'output': nsd_dict['cap'], 'c_crossattn': instruction_text, 'c_crossattn_1': fmri_norm}
                nsd_dict['edited'] = init_image[0]
        # print(nsd_dict['cap'], nsd_dict['fmri_edit']['c_crossattn'])
        return nsd_dict

    def __len__(self):
        return len(self.cocos)


if __name__ == '__main__':
    dataset = NIPS23NSDDataset()
    for i, in_dict in enumerate(dataset):
        # print(in_dict.keys())
        # print(in_dict[''])
        if i>20: import pdb; pdb.set_trace();