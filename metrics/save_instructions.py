import os
import sys
import argparse
import glob
from PIL import Image
import torchvision.transforms as transforms
import torchvision
from clip_similarity import ClipSimilarity
from dino_similarity import DINO_v2_Similarity

import torch
from tqdm import tqdm
import torch.nn.functional as F
import heapq
import pickle


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--eval_method', type=str, default='clip')
    parser.add_argument('--root_dir', type=str, default='logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_sdedit')
    parser.add_argument('--text_dir', type=str, default='logs/test_conv_adaptor_test_conv_adaptor_2024-04-13/visualize/images/val')
    parser.add_argument('--eval_mode', type=str, default='id_sim')
    args = parser.parse_args()

    if args.eval_mode == 'id_sim':
        pass
    elif args.eval_mode == 'inst_sim':
        args.eval_method = 'clip'
    elif args.eval_method == 'fid':
        pass
    else:
        raise ValueError

    if args.eval_method == 'clip':
        sim_evaluator = ClipSimilarity().cuda()
    elif args.eval_method == 'dino':
        sim_evaluator = DINO_v2_Similarity().cuda()
    elif args.eval_method == 'fid':
        pass
    else:
        raise ValueError

    img_paths = sorted(glob.glob(os.path.join(args.root_dir, 'all_iter*.png')))
    # img_paths = img_paths[:15]
    # print(img_paths)

    key_words = ['make', 'replace', 'turn', 'change', 'remove', 'add', 'insert', 'swap', 'switch', 'put', 'cut']
    id_sim_dict = {key_word: [] for key_word in key_words}
    id_sim_dict['other'] = []

    all_id_sims, effective_img_paths = [], []
    uneffective_img_paths = []
    input_imgs, edit_imgs = [], []

    # if os.path.exists('uneffective_img_paths.pkl'):
    #     with open('uneffective_img_paths.pkl', 'rb') as f:
    #         uneffective_img_paths = pickle.load(f)

    instruct_texts, input_texts = [], []
    for i,img_path in tqdm(enumerate(img_paths)):
        instruct_path = img_path.replace(args.root_dir, args.text_dir).replace('.png', '-instruct.txt')
        with open(instruct_path, 'r') as f:
            instruct_text = f.readlines()
        instruct_texts += instruct_text
        
        input_path = img_path.replace(args.root_dir, args.text_dir).replace('.png', '-input.txt')
        with open(input_path, 'r') as f:
            input_text = f.readlines()
        input_texts += input_text
    # import pdb; pdb.set_trace()

    # instruction_out_path = 'instruction_out.txt'
    # with open(instruction_out_path, 'w') as f:
    #     for i, instruction in enumerate(instruct_texts):
    #         f.write(instruction+'\n')
    
    input_instruction_out_path = 'input_instruction_out.txt'
    with open(input_instruction_out_path, 'w') as f:
        for i, (input_text, instruction) in enumerate(zip(input_texts, instruct_texts)):
            print(instruction)
            import pdb; pdb.set_trace()
            f.write(input_text + '  ' + instruction+'\n')
    

if __name__ == '__main__':
    main()
