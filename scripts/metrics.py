#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# # Code to convert this notebook to .py if you want to run it via command line or with Slurm
# from subprocess import call
# command = "jupyter nbconvert Reconstruction_Metrics.ipynb --to python"
# call(command,shell=True)


# In[2]:


import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from datetime import datetime
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
local_rank = 0
print("device:",device)

import utils

recon_path = "./recon_path2-981/"
all_images_path  = "./all_images_path2-981/"
seed=42
num = 100 #980
imsize = 512



# In[6]:
from PIL import Image

def load_image_as_tensor(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    image = transform(image)
    image_numpy = np.array(image)
    image_tensor = torch.from_numpy(image_numpy).unsqueeze(0)  # 添加一个维度以表示batch size
    return image_tensor

def stack_images_in_folder(folder_path):
    images_tensor_list = []
    number=0
    for filename in os.listdir(folder_path):
        if filename.endswith(".png") and (number< (num+1)) :  # 只处理PNG文件
            print(number)
            image_path = os.path.join(folder_path, filename)
            image_tensor = load_image_as_tensor(image_path)
            images_tensor_list.append(image_tensor)
            number = number+1
    stacked_images_tensor = torch.cat(images_tensor_list, dim=0)  # 在batch dimension上进行堆叠
    return stacked_images_tensor

all_brain_recons = stack_images_in_folder(recon_path)
all_images =  stack_images_in_folder(all_images_path)



# def load_image_as_tensor2(image_path):
#     image = Image.open(image_path).convert('RGB')
#     # transform = transforms.ToTensor()
#     # image = transform(image)
#     image_numpy = np.array(image)
#     image_tensor = torch.from_numpy(image_numpy).unsqueeze(0)  # 添加一个维度以表示batch size
#     return image_tensor

# def stack_images_in_folder2(folder_path):
#     images_tensor_list = []
#     number=0
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".png") and (number< (num+1)) :  # 只处理PNG文件
#             print(number)
#             image_path = os.path.join(folder_path, filename)
#             image_tensor = load_image_as_tensor2(image_path)
#             images_tensor_list.append(image_tensor)
#             number = number+1
#     stacked_images_tensor = torch.cat(images_tensor_list, dim=0)  # 在batch dimension上进行堆叠
#     return stacked_images_tensor


# all_brain_recons2 = stack_images_in_folder2(recon_path).permute(0,3,1,2).float()
# all_images2 =  stack_images_in_folder2(all_images_path).permute(0,3,1,2).float()


print(all_images.shape)
print(all_brain_recons.shape)

all_images = all_images.to(device)
all_brain_recons = all_brain_recons.to(device).to(all_images.dtype).clamp(0,1)


# # Display reconstructions next to ground truth images

# In[8]:


imsize = 256
all_images = transforms.Resize((imsize,imsize))(all_images)
all_brain_recons = transforms.Resize((imsize,imsize))(all_brain_recons)

np.random.seed(0)
ind = np.flip(np.array([112,119,101,44,159,22,173,174,175,189,981,243,249,255,265]))

# all_interleaved = torch.zeros(len(ind)*2,3,imsize,imsize)
# icount = 0
# for t in ind:
#     all_interleaved[icount] = all_images[t]
#     all_interleaved[icount+1] = all_brain_recons[t]
#     icount += 2

# plt.rcParams["savefig.bbox"] = 'tight'
# def show(imgs,figsize):
#     if not isinstance(imgs, list):
#         imgs = [imgs]
#     fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=figsize)
#     for i, img in enumerate(imgs):
#         img = img.detach()
#         img = transforms.ToPILImage()(img)
#         axs[0, i].imshow(np.asarray(img))
#         axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    
# grid = make_grid(all_interleaved, nrow=10, padding=2)
# show(grid,figsize=(20,16))


# # 2-Way Identification

# In[9]:


from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

@torch.no_grad()
def two_way_identification(all_brain_recons, all_images, model, preprocess, feature_layer=None, return_avg=True):
    preds = model(torch.stack([preprocess(recon) for recon in all_brain_recons], dim=0).to(device))
    reals = model(torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device))
    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()

    r = np.corrcoef(reals, preds)
    r = r[:len(all_images), len(all_images):]
    congruents = np.diag(r)

    success = r < congruents
    success_cnt = np.sum(success, 0)

    if return_avg:
        perf = np.mean(success_cnt) / (len(all_images)-1)
        return perf
    else:
        return success_cnt, len(all_images)-1


# ## PixCorr

# In[10]:


preprocess = transforms.Compose([
    transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
])

# Flatten images while keeping the batch dimension
all_images_flattened = preprocess(all_images.cpu()).reshape(len(all_images), -1).cpu()
all_brain_recons_flattened = preprocess(all_brain_recons.cpu()).view(len(all_brain_recons), -1).cpu()

print(all_images_flattened.shape)
print(all_brain_recons_flattened.shape)

corrsum = 0
for i in tqdm(range(num)):
    corrsum += np.corrcoef(all_images_flattened[i], all_brain_recons_flattened[i])[0][1]
corrmean = corrsum / num

pixcorr = corrmean
print("pixcorr: ",pixcorr)


# ## SSIM

# In[11]:


# see https://github.com/zijin-gu/meshconv-decoding/issues/3
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim

preprocess = transforms.Compose([
    transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR), 
])

# convert image to grayscale with rgb2grey
img_gray = rgb2gray(preprocess(all_images).permute((0,2,3,1)).cpu())
recon_gray = rgb2gray(preprocess(all_brain_recons).permute((0,2,3,1)).cpu())
print("converted, now calculating ssim...")

ssim_score=[]
for im,rec in tqdm(zip(img_gray,recon_gray),total=len(all_images)):
    ssim_score.append(ssim(rec, im, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0))

ssim = np.mean(ssim_score)
print("ssim:  ", ssim)


# ### AlexNet

# In[12]:


from torchvision.models import alexnet, AlexNet_Weights
alex_weights = AlexNet_Weights.IMAGENET1K_V1

alex_model = create_feature_extractor(alexnet(weights=alex_weights), return_nodes=['features.4','features.11']).to(device)
alex_model.eval().requires_grad_(False)

# see alex_weights.transforms()
preprocess = transforms.Compose([
    transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

layer = 'early, AlexNet(2)'
print(f"\n---{layer}---")
all_per_correct = two_way_identification(all_brain_recons.to(device).float(), all_images, 
                                                          alex_model, preprocess, 'features.4')
alexnet2 = np.mean(all_per_correct)
print(f"2-way Percent Correct: {alexnet2:.4f}")

layer = 'mid, AlexNet(5)'
print(f"\n---{layer}---")
all_per_correct = two_way_identification(all_brain_recons.to(device).float(), all_images, 
                                                          alex_model, preprocess, 'features.11')
alexnet5 = np.mean(all_per_correct)
print(f"2-way Percent Correct: {alexnet5:.4f}")


# ### InceptionV3

# In[13]:


from torchvision.models import inception_v3, Inception_V3_Weights
weights = Inception_V3_Weights.DEFAULT
inception_model = create_feature_extractor(inception_v3(weights=weights), 
                                           return_nodes=['avgpool']).to(device)
inception_model.eval().requires_grad_(False)

# see weights.transforms()
preprocess = transforms.Compose([
    transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

all_per_correct = two_way_identification(all_brain_recons, all_images,
                                        inception_model, preprocess, 'avgpool')
        
inception = np.mean(all_per_correct)
print(f"2-way Percent Correct: {inception:.4f}")


# # ### CLIP

# # In[14]:


# import clip
# clip_model, preprocess = clip.load("ViT-L/14", device=device)

# preprocess = transforms.Compose([
#     transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
#     transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
#                          std=[0.26862954, 0.26130258, 0.27577711]),
# ])

# all_per_correct = two_way_identification(all_brain_recons, all_images,
#                                         clip_model.encode_image, preprocess, None) # final layer
# clip_ = np.mean(all_per_correct)
# print(f"2-way Percent Correct: {clip_:.4f}")


# ### Efficient Net

# In[15]:


from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
weights = EfficientNet_B1_Weights.DEFAULT
eff_model = create_feature_extractor(efficientnet_b1(weights=weights), 
                                    return_nodes=['avgpool']).to(device)
eff_model.eval().requires_grad_(False)

# see weights.transforms()
preprocess = transforms.Compose([
    transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

gt = eff_model(preprocess(all_images))['avgpool']
gt = gt.reshape(len(gt),-1).cpu().numpy()
fake = eff_model(preprocess(all_brain_recons))['avgpool']
fake = fake.reshape(len(fake),-1).cpu().numpy()

effnet = np.array([sp.spatial.distance.correlation(gt[i],fake[i]) for i in range(len(gt))]).mean()
print("Distance:",effnet)


# ### SwAV

# In[16]:


swav_model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
swav_model = create_feature_extractor(swav_model, 
                                    return_nodes=['avgpool']).to(device)
swav_model.eval().requires_grad_(False)

preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

gt = swav_model(preprocess(all_images))['avgpool']
gt = gt.reshape(len(gt),-1).cpu().numpy()
fake = swav_model(preprocess(all_brain_recons))['avgpool']
fake = fake.reshape(len(fake),-1).cpu().numpy()

swav = np.array([sp.spatial.distance.correlation(gt[i],fake[i]) for i in range(len(gt))]).mean()
print("Distance:",swav)


# # Display in table

# In[34]:


# Create a dictionary to store variable names and their corresponding values
data = {
    "Metric": ["PixCorr", "SSIM", "AlexNet(2)", "AlexNet(5)", "InceptionV3", "EffNet-B", "SwAV"],
    "Value": [pixcorr, ssim, alexnet2, alexnet5, inception, effnet, swav],
}

df = pd.DataFrame(data)
print(df.to_string(index=False))

 