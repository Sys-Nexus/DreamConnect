import os
import sys
import argparse
import glob
from PIL import Image
import torchvision.transforms as transforms
import torchvision
from clip_similarity import ClipSimilarity
from dino_similarity import DINO_v2_Similarity

from tqdm import tqdm
import torch.nn.functional as F


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
    # parser.add_argument('--root_dir', type=str, default='logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_inst_pix2pix')
    # parser.add_argument('--root_dir', type=str, default='logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_inst_dif')
    parser.add_argument('--root_dir', type=str, default='logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_sdedit')
    parser.add_argument('--text_dir', type=str, default='logs/test_conv_adaptor_test_conv_adaptor_2024-04-13/visualize/images/val')
    parser.add_argument('--eval_mode', type=str, default='id_sim')
    # parser.add_argument('--root_dir', type=str, default='logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_magic_brush')
    args = parser.parse_args()

    if args.eval_mode == 'id_sim':
        pass
    elif args.eval_mode == 'inst_sim':
        args.eval_method = 'clip'
    else:
        raise ValueError

    if args.eval_method == 'clip':
        sim_evaluator = ClipSimilarity().cuda()
    elif args.eval_method == 'dino':
        sim_evaluator = DINO_v2_Similarity().cuda()
    else:
        raise ValueError

    img_paths = sorted(glob.glob(os.path.join(args.root_dir, 'all_iter*.png')))
    img_paths = img_paths[:6]
    print(img_paths)
    all_id_sims = []
    for i,img_path in tqdm(enumerate(img_paths)):
        input_img = read_split_image(img_path, 3)
        edit_img = read_split_image(img_path, 4)
        input_feat = sim_evaluator.encode_image(input_img)
        edit_feat = sim_evaluator.encode_image(edit_img)
        if args.eval_mode == 'id_sim':
            id_sim = F.cosine_similarity(input_feat, edit_feat)
        elif args.eval_mode == 'inst_sim':
            input_path = img_path.replace(args.root_dir, args.text_dir).replace('.png', '-input.txt')
            with open(input_path, 'r') as f:
                input_text = f.readlines()
            output_path = img_path.replace(args.root_dir, args.text_dir).replace('.png', '-output.txt')
            with open(output_path, 'r') as f:
                output_text = f.readlines()
            instruct_path = img_path.replace(args.root_dir, args.text_dir).replace('.png', '-instruct.txt')
            with open(instruct_path, 'r') as f:
                instruct_text = f.readlines()

            # instr_text = ['']
            # import pdb; pdb.set_trace()
            # instr_feat = sim_evaluator.encode_text(instr_text)
            instr_feat = sim_evaluator.encode_text(output_text) - sim_evaluator.encode_text(input_text)
            delta_feat = edit_feat - input_feat
            id_sim = F.cosine_similarity(delta_feat / delta_feat.norm(dim=1, keepdim=True), instr_feat)
        else:
            raise ValueError
        all_id_sims.append(id_sim.item())
        # if i>5:break
    ave_id_sim = sum(all_id_sims) * 1.0 / len(all_id_sims)
    print('eval_mode: ', args.eval_mode)
    print('root_dir: ', args.root_dir)
    print('id_sim: ', ave_id_sim)

    # import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
