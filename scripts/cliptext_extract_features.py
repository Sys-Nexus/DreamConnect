import torch
import numpy as np
import os
import sys
from easydict import EasyDict

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_root)
sys.path.append(os.path.join(proj_root, "stable_diffusion"))
sys.path.append(os.path.join(proj_root, 'third_party/versatile_diffusion'))

from lib.model_zoo.vd import VDCLIP
from third_party.StableDiffusionReconstruction.codes.utils.nsd_creater import NIPS23NSDDataset


def main():
    sub = 1
    # train_caps = np.load('data/processed_data/subj{:02d}/nsd_train_cap_sub{}.npy'.format(sub,sub)) 
    # test_caps = np.load('data/processed_data/subj{:02d}/nsd_test_cap_sub{}.npy'.format(sub,sub))  

    # cfgm_name = 'vd_noema'
    # pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
    # cfgm = model_cfg_bank()(cfgm_name)
    # net = get_model()(cfgm)

    # sd = torch.load(pth, map_location='cpu')
    # net.load_state_dict(sd, strict=False)  
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # net.clip = net.clip.to(device)
    
    # params copied from configs/versatile_dualstream.yaml
    clip_params = {'symbol':'clip', 'args':{}, 'name':'clip_frozen', 'type':'clip_frozen'}
    clip_cfg = EasyDict(clip_params)
    vd_clip = VDCLIP(clip_cfg)
    
    train_ds_params = {
        'nsd_root': '/data/yashengsun/Proj/MMEdit/StableDiffusionReconstruction/nsd',
        'resolution': 320,
        'split': 'train',
        'is_reconstruct_mode': False,
        'url': 'nsd_data_dir/train_subj01_{0..17}.tar',
        'reconstruct_prob': 1.01,
    }
    test_ds_params = {
        'nsd_root': '/data/yashengsun/Proj/MMEdit/StableDiffusionReconstruction/nsd',
        'resolution': 320,
        'split': 'test',
        'is_reconstruct_mode': False,
        'url': 'nsd_data_dir/test_subj01_{0..1}.tar',
        'reconstruct_prob': 1.01,
    }
    
    train_dataset = NIPS23NSDDataset(**train_ds_params)
    test_dataset = NIPS23NSDDataset(**test_ds_params)

    num_embed, num_features, num_test, num_train = 77, 768, len(test_dataset), len(train_dataset)

    train_clip = np.zeros((num_train, num_embed, num_features))
    test_clip = np.zeros((num_test, num_embed, num_features))

    os.makedirs('nsd_data_dir/extracted_features/subj{:02d}'.format(sub), exist_ok=True)
    # import pdb; pdb.set_trace()
    with torch.no_grad():
        for i in range(num_test):
            cin = [test_dataset[i]['cap']]
            c = vd_clip.clip_encode_text(cin)
            test_clip[i] = c.to('cpu').numpy().mean(0)
        
        np.save('nsd_data_dir/extracted_features/subj{:02d}/nsd_cliptext_test.npy'.format(sub),test_clip)
            
        for i in range(num_train):
            cin = [test_dataset[i]['cap']]
            c = vd_clip.clip_encode_text(cin)
            train_clip[i] = c.to('cpu').numpy().mean(0)
        np.save('nsd_data_dir/extracted_features/subj{:02d}/nsd_cliptext_train.npy'.format(sub),train_clip)


if __name__ == '__main__':
    main()