import os
import sys
import argparse
import glob
from PIL import Image
import torchvision.transforms as transforms
from clip_similarity import ClipSimilarity



def read_split_image(img_path, offset):
    img = Image.open(img_path)
    import pdb; pdb.set_trace()
    img = img.crop((offset, 0, offset + 256, 256))
    img_ts = transforms.ToTensor()(img)
    return img_ts


def main():
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--root_dir', type=str, default='logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_inst_pix2pix')
    # parser.add_argument('--root_dir', type=str, default='logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_inst_dif')
    parser.add_argument('--root_dir', type=str, default='logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_sdedit')
    # parser.add_argument('--root_dir', type=str, default='logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_magic_brush')
    args = parser.parse_args()

    sim_evaluator = ClipSimilarity().cuda()
    img_paths = glob.glob(os.path.join(args.root_dir, 'all_iter*.png'))
    all_id_sims = []
    for img_path in img_paths:
        input_img = read_split_image(img_path, 3)
        edit_img = read_split_image(img_path, 4)
        input_feat = sim_evaluator.encode_image(input_img)
        edit_feat = sim_evaluator.encode_image(edit_img)
        id_sim = F.cosine_similarity(input_feat, edit_feat)
        all_id_sims.append(id_sim.item())
    ave_id_sim = sum(all_id_sims) / len(all_id_sims)
    print('id_sim: ', ave_id_sim)

    # import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
