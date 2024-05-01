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
# from instruction_entity import instruction_entity_dict


def exe_sam(text_prompt, image_path):
    # cd_cmd = 'cd /data/yashengsun/Proj/MMEdit/Grounded-Segment-Anything'
    # print(cd_cmd)
    # os.system(cd_cmd)

    # text_prompt = 'cat'
    # image_path = '/data/yashengsun/Proj/Diffusion/InstructDiffusion/teaser_samples/all_iter-999999_ep-999999_bidx-000234-020064.png'
    output_dir = 'outputs'
    exe_cmd = 'python /data/yashengsun/Proj/MMEdit/Grounded-Segment-Anything/grounded_sam_demo.py --config /data/yashengsun/Proj/MMEdit/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --grounded_checkpoint /data/yashengsun/Proj/MMEdit/Grounded-Segment-Anything/groundingdino_swint_ogc.pth   --sam_checkpoint /data/yashengsun/Proj/MMEdit/Grounded-Segment-Anything/sam_vit_h_4b8939.pth   --input_image {} --output_dir {} --box_threshold 0.3 --text_threshold 0.25 --text_prompt {} --device cuda'.format(image_path, output_dir, text_prompt)
    print(exe_cmd)
    os.system(exe_cmd)

    # cd_cmd = 'cd /data/yashengsun/Proj/MMEdit/MindSculpt'
    # print(cd_cmd)
    # os.system(cd_cmd)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--eval_method', type=str, default='clip')
    parser.add_argument('--root_dir', type=str, default='/data/yashengsun/Proj/MMEdit/MindSculpt/logs/test_res_value_inject_idback_css15_test_res_value_inject_idback_css15_2024-04-25/visualize/images/val')
    parser.add_argument('--text_dir', type=str, default='/data/yashengsun/Proj/MMEdit/MindSculpt/logs/test_conv_adaptor_test_conv_adaptor_2024-04-13/visualize/images/val')
    parser.add_argument('--eval_mode', type=str, default='id_sim')
    args = parser.parse_args()

    img_paths = sorted(glob.glob(os.path.join(args.root_dir, 'all_iter*.png')))
    obtained_entitites = []
    with open('./obtained_areas.txt', 'r') as f:
        obtained_entitites = f.readlines()
        obtained_entitites = [x.strip() for x in obtained_entitites]
    num_entitites = len(obtained_entitites)
    img_paths = img_paths[:num_entitites]

    for i,(img_path, obtained_entity) in tqdm(enumerate(zip(img_paths, obtained_entitites))):
        instruct_path = img_path.replace(args.root_dir, args.text_dir).replace('.png', '-instruct.txt')
        obtained_entity = obtained_entity.replace('.','')
        with open(instruct_path, 'r') as f:
            instruct_text = f.readlines()[0]
        # if instruct_text not in instruction_entity_dict: continue
        if 'add' in instruct_text.lower() or 'put' in instruct_text.lower() or 'insert' in instruct_text.lower(): continue
        if 'entire' in obtained_entity.lower() or 'specified' in obtained_entity.lower(): continue
        trim_img_path = img_path.replace('/val', '/val_trimmed')
        if (i+3) % 100 == 0 or (i+2) % 100 == 0 or (i+1) % 100 == 0 or (i) % 100 == 0:
            print(i, instruct_text, obtained_entity)
        # exe_sam(entity, trim_img_path)
        # import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
    # mask_path = '/data/yashengsun/Proj/MMEdit/Grounded-Segment-Anything/outputs/mask.jpg'
    # mask = cv2.imread(mask_path)
    # print(np.unique(mask))
    # print(mask.shape)
