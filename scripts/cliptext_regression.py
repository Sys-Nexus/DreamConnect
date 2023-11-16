import sys
import numpy as np
import sklearn.linear_model as skl
import pickle
import argparse
import torch
import numpy as np
import os
import sys
from easydict import EasyDict
from tqdm import tqdm

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_root)
sys.path.append(os.path.join(proj_root, "stable_diffusion"))
sys.path.append(os.path.join(proj_root, 'third_party/versatile_diffusion'))

from lib.model_zoo.vd import VDCLIP
from third_party.StableDiffusionReconstruction.codes.utils.nsd_creater import NIPS23NSDDataset


def main():
    sub = 1

    train_path = 'nsd_data_dir/processed_data/subj{:02d}/nsd_train_fmriavg_nsdgeneral_sub{}.npy'.format(sub,sub)
    train_fmri = np.load(train_path).astype(np.float32)
    test_path = 'nsd_data_dir/processed_data/subj{:02d}/nsd_test_fmriavg_nsdgeneral_sub{}.npy'.format(sub,sub)
    test_fmri = np.load(test_path).astype(np.float32)
    print(np.mean(train_fmri),np.std(train_fmri))
    print(np.mean(test_fmri),np.std(test_fmri))
    # import pdb; pdb.set_trace();

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
    import pdb; pdb.set_trace()

    #### TODO: add preprocessing normalization.
    num_voxels, num_train, num_test = train_fmri.shape[1], len(train_fmri), len(test_fmri)

    train_clip = np.load('nsd_data_dir/extracted_features/subj{:02d}/nsd_cliptext_train.npy'.format(sub))
    test_clip = np.load('nsd_data_dir/extracted_features/subj{:02d}/nsd_cliptext_test.npy'.format(sub))

    train_path = 'nsd_data_dir/processed_data/subj{:02d}/nsd_train_fmriavg_nsdgeneral_sub{}.npy'.format(sub,sub)
    train_fmri = np.load(train_path)
    test_path = 'nsd_data_dir/processed_data/subj{:02d}/nsd_test_fmriavg_nsdgeneral_sub{}.npy'.format(sub,sub)
    test_fmri = np.load(test_path)

    ## Regression
    num_samples,num_embed,num_dim = train_clip.shape

    print("Training Regression")
    reg_w = np.zeros((num_embed,num_dim,num_voxels)).astype(np.float32)
    reg_b = np.zeros((num_embed,num_dim)).astype(np.float32)
    pred_clip = np.zeros_like(test_clip)
    
    os.makedirs('nsd_data_dir/predicted_features/subj{:02d}'.format(sub), exist_ok=True)
    os.makedirs('nsd_data_dir/regression_weights/subj{:02d}'.format(sub), exist_ok=True)
    for i in tqdm(range(num_embed)):
        reg = skl.Ridge(alpha=100000, max_iter=50000, fit_intercept=True)
        reg.fit(train_fmri, train_clip[:,i])
        reg_w[i] = reg.coef_
        reg_b[i] = reg.intercept_
        
        pred_test_latent = reg.predict(test_fmri)
        # std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent,axis=0)) / np.std(pred_test_latent,axis=0)
        # pred_clip[:,i] = std_norm_test_latent * np.std(train_clip[:,i],axis=0) + np.mean(train_clip[:,i],axis=0)
        pred_clip[:,i] = pred_test_latent
        print(i,reg.score(test_fmri,test_clip[:,i]))

    np.save('nsd_data_dir/predicted_features/subj{:02d}/nsd_cliptext_predtest_nsdgeneral.npy'.format(sub),pred_clip)


    datadict = {
        'weight' : reg_w,
        'bias' : reg_b,

    }

    with open('nsd_data_dir/regression_weights/subj{:02d}/cliptext_regression_weights.pkl'.format(sub),"wb") as f:
        pickle.dump(datadict,f)


if __name__ == '__main__':
    main()