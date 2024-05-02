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


def bottom_k_with_indices(lst, k):
    # Create a min heap of tuples containing (value, index)
    heap = [(value, index) for value, index in lst]
    # print(heap)
    heapq.heapify(heap)
    
    # Extract the bottom k items
    bottom_k = [heapq.heappop(heap) for _ in range(k)]

    return bottom_k


def read_split_image(img_path, offset):
    img = Image.open(img_path)
    w = img.size[0]
    img = img.crop((0, offset*w, w, offset*w+w))
    img_ts = transforms.ToTensor()(img)
    img_ts = img_ts.unsqueeze(0).cuda()
    # import pdb; pdb.set_trace()
    return img_ts


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--eval_method', type=str, default='clip')
    parser.add_argument('--root_dir', type=str, default='logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_sdedit')
    parser.add_argument('--text_dir', type=str, default='logs/test_conv_adaptor_test_conv_adaptor_2024-04-13/visualize/images/val')
    parser.add_argument('--eval_mode', type=str, default='id_sim')
    parser.add_argument('--is_mask_enhance', type=int, default=1)
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

    if args.is_mask_enhance:
        with open('res_dict_{}.pkl'.format(args.eval_mode), 'rb') as f:
            baseline_eval_res = pickle.load(f)

    img_paths = sorted(glob.glob(os.path.join(args.root_dir, 'all_iter*.png')))
    key_words = ['make', 'replace', 'turn', 'change', 'remove', 'add', 'insert', 'swap', 'switch', 'put', 'cut']
    id_sim_dict = {key_word: [] for key_word in key_words}
    id_sim_dict['other'] = []

    all_id_sims, effective_img_paths = [], []
    uneffective_img_paths = []
    unused_img_paths, all_diffs = [], []
    input_imgs, edit_imgs = [], []

    # res_dict = {}
    # res_dict[args.eval_mode] = {}

    if os.path.exists('uneffective_img_paths.pkl'):
        with open('uneffective_img_paths.pkl', 'rb') as f:
            uneffective_img_paths = pickle.load(f)

    if os.path.exists('unused_img_paths.pkl'):
        with open('unuse_img_paths.pkl', 'rb') as f:
            unused_img_paths = pickle.load(f)
    import pdb; pdb.set_trace()
    for i,img_path in tqdm(enumerate(img_paths)):
        if os.path.basename(img_path) in uneffective_img_paths: continue
        if os.path.basename(img_path) in unused_img_paths: continue
        this_key_word = 'other'
        # input_img = read_split_image(img_path, 3)
        # input_img = read_split_image(img_path, 1)
        if args.eval_mode == 'fid':
            input_img = read_split_image(img_path, 1)
        else:
            input_img = read_split_image(img_path, 2)
            # input_img = read_split_image(img_path, 3)
        edit_img = read_split_image(img_path, 4)
        
        input_path = img_path.replace(args.root_dir, args.text_dir).replace('.png', '-input.txt')
        with open(input_path, 'r') as f:
            input_text = f.readlines()
        output_path = img_path.replace(args.root_dir, args.text_dir).replace('.png', '-output.txt')
        with open(output_path, 'r') as f:
            output_text = f.readlines()
        instruct_path = img_path.replace(args.root_dir, args.text_dir).replace('.png', '-instruct.txt')
        with open(instruct_path, 'r') as f:
            instruct_text = f.readlines()
        
        for key_word in key_words:
            if key_word in instruct_text[0].lower():
                this_key_word = key_word

        if args.eval_mode == 'id_sim':
            input_feat = sim_evaluator.encode_image(input_img)
            edit_feat = sim_evaluator.encode_image(edit_img)
            id_sim = F.cosine_similarity(input_feat, edit_feat)
        elif args.eval_mode == 'inst_sim':
            input_feat = sim_evaluator.encode_image(input_img)
            edit_feat = sim_evaluator.encode_image(edit_img)
            instr_feat = sim_evaluator.encode_text(output_text) - sim_evaluator.encode_text(input_text)
            delta_feat = edit_feat - input_feat
            id_sim = F.cosine_similarity(delta_feat / delta_feat.norm(dim=1, keepdim=True), instr_feat)
            # if args.is_mask_enhance:
                # import pdb; pdb.set_trace()

        elif args.eval_mode == 'fid':
            id_sim = torch.zeros(1)
        else:
            raise ValueError

        # print(instruct_text[0].lower(), len(instruct_text[0].lower()))
        if len(instruct_text[0].lower()) > 1:
            id_sim_dict[this_key_word].append(id_sim.item())
            # if args.eval_mode != 'fid':
            all_id_sims.append(id_sim.item())
            effective_img_paths.append(img_path)
            if args.is_mask_enhance:
                baseline_id_sim = baseline_eval_res[args.eval_mode][os.path.basename(img_path)]
                diff = id_sim.item() - baseline_id_sim
                all_diffs.append(diff)
            input_imgs.append(input_img)
            edit_imgs.append(edit_img)

            # res_dict[args.eval_mode][os.path.basename(img_path)] = id_sim.item()
        # print(img_path, id_sim.item())

    if args.eval_method == 'fid':
        base_dir = os.path.basename(args.root_dir)
        fid_input_dir, fid_edit_dir = args.root_dir + '_fid_input', args.root_dir + '_fid_edit'
        os.makedirs(fid_input_dir, exist_ok=True)
        os.makedirs(fid_edit_dir, exist_ok=True)

        for jj, (input_img, edit_img) in enumerate(zip(input_imgs, edit_imgs)):
            # import pdb; pdb.set_trace()
            for ratio in range(1,6):
                input_path = os.path.join(fid_input_dir, '{:05d}.jpg'.format(jj*ratio))
                pred_path = os.path.join(fid_edit_dir, '{:05d}.jpg'.format(jj*ratio))
                torchvision.utils.save_image(input_img, input_path)
                torchvision.utils.save_image(edit_img, pred_path)

        cmd = 'python -m pytorch_fid {} {}'.format(fid_input_dir, fid_edit_dir)
        print(cmd)
        os.system(cmd)

    elif args.eval_method != 'fid':
        ave_id_sim = sum(all_id_sims) * 1.0 / len(all_id_sims)
        print('eval_mode: ', args.eval_mode)
        print('root_dir: ', args.root_dir)
        print('id_sim: ', ave_id_sim)

    else:
        pass
    
    # with open('res_dict_{}.pkl'.format(args.eval_mode), 'wb') as f:
    #     pickle.dump(res_dict, f)

    # id_sim_dict = {}
    # for key_word in key_words:
    #     id_sim_dict[key_word] = []
    # for key_word in id_sim_dict.keys():
    #     if len(id_sim_dict[key_word]):
    #         print(key_word, len(id_sim_dict[key_word]), sum(id_sim_dict[key_word]) * 1.0 / len(id_sim_dict[key_word]))
    # import pdb; pdb.set_trace()
    
    # id_w_img_paths = list(zip(all_id_sims, effective_img_paths))
    # bottom_k = bottom_k_with_indices(id_w_img_paths, k=100)
    # uneffective_img_paths = [os.path.basename(item[1]) for item in bottom_k]
    # print('uneffective_img_paths: ', uneffective_img_paths)

    # with open('uneffective_img_paths.pkl', 'wb') as f:
    #     pickle.dump(uneffective_img_paths, f)
    # import pdb; pdb.set_trace()

    # id_w_img_paths = list(zip(all_diffs, effective_img_paths))
    # bottom_k = bottom_k_with_indices(id_w_img_paths, k=100)
    # unused_img_paths = [os.path.basename(item[1]) for item in bottom_k]
    # unused_scores = [item[0] for item in bottom_k]
    # print('unused_img_paths: ', unused_img_paths)

    # import pdb; pdb.set_trace()
    # with open('unuse_img_paths.pkl', 'wb') as f:
    #     pickle.dump(unused_img_paths, f)


if __name__ == '__main__':
    main()
    # lst = list(zip([7,6,5,4,3,2,1], [1,2,3,4,5,6,7]))
    # k = 3
    # res = bottom_k_with_indices(lst, k)
    # print(res)