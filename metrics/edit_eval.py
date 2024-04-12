import os
import sys
import argparse
import glob
from clip_similarity import ClipSimilarity



def main():
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--root_dir', type=str, default='logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_inst_pix2pix')
    # parser.add_argument('--root_dir', type=str, default='logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_inst_dif')
    parser.add_argument('--root_dir', type=str, default='logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_sdedit')
    # parser.add_argument('--root_dir', type=str, default='logs/test_conv_adaptor_test_conv_adaptor_2024-04-08/visualize/images/val_magic_brush')
    args = parser.parse_args()

    sim_evaluator = ClipSimilarity().cuda()

    img_paths = glob.glob(os.path.join(args.root_dir, 'all_iter*.png'))
    import pdb; pdb.set_trace()
    

if __name__ == '__main__':
    main()
