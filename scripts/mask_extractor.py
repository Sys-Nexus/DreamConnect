import os
import cv2
import numpy as np
import os
import sys
import argparse
import glob
from PIL import Image
import torchvision.transforms as transforms
import torchvision

import torch
from tqdm import tqdm
import torch.nn.functional as F
import heapq
import pickle
from instruction_entity import instruction_entity_dict


def read_sam(text_prompt, image_path):
    cd_cmd = 'cd ~/Proj/MMEdit/Grounded-Segment-Anything'
    print(cd_cmd)
    os.system(cd_cmd)

    # text_prompt = 'cat'
    # image_path = '/data/yashengsun/Proj/Diffusion/InstructDiffusion/teaser_samples/all_iter-999999_ep-999999_bidx-000234-020064.png'
    output_dir = 'outputs'
    exe_cmd = 'python grounded_sam_demo.py --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --grounded_checkpoint groundingdino_swint_ogc.pth   --sam_checkpoint sam_vit_h_4b8939.pth   --input_image {} --output_dir {} --box_threshold 0.3 --text_threshold 0.25 --text_prompt {} --device cuda'.format(image_path, output_dir, text_prompt)
    print(exe_cmd)
    os.system(exe_cmd)

    cd_cmd = 'cd ~/Proj/MMEdit/MindSculpt'
    print(cd_cmd)
    os.system(cd_cmd)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--eval_method', type=str, default='clip')
    parser.add_argument('--root_dir', type=str, default='logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_sdedit')
    parser.add_argument('--text_dir', type=str, default='logs/test_conv_adaptor_test_conv_adaptor_2024-04-13/visualize/images/val')
    parser.add_argument('--eval_mode', type=str, default='id_sim')
    args = parser.parse_args()

    img_paths = sorted(glob.glob(os.path.join(args.root_dir, 'all_iter*.png')))
    
    for i,img_path in tqdm(enumerate(img_paths)):
        instruct_path = img_path.replace(args.root_dir, args.text_dir).replace('.png', '-instruct.txt')
        with open(instruct_path, 'r') as f:
            instruct_text = f.readlines()[0]
        entity = instruction_entity_dict[instruct_text]
        import pdb; pdb.set_trace()



if __name__ == '__main__':
    main()
    # mask_path = '/data/yashengsun/Proj/MMEdit/Grounded-Segment-Anything/outputs/mask.jpg'
    # mask = cv2.imread(mask_path)
    # print(np.unique(mask))
    # print(mask.shape)
