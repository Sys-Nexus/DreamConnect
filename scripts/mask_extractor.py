import os
import cv2
import numpy as np

def main():

    cd_cmd = 'cd ~/Proj/MMEdit/Grounded-Segment-Anything'
    os.system(cd_cmd)

    env_cmd = 'conda activate tformer4192'
    os.system(env_cmd)

    text_prompt = 'cat'
    image_path = '/data/yashengsun/Proj/Diffusion/InstructDiffusion/teaser_samples/all_iter-999999_ep-999999_bidx-000234-020064.png'
    output_dir = 'output'
    exe_cmd = 'python grounded_sam_demo.py --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --grounded_checkpoint groundingdino_swint_ogc.pth   --sam_checkpoint sam_vit_h_4b8939.pth   --input_image {} --output_dir {} --box_threshold 0.3 --text_threshold 0.25 --text_prompt {} --device cuda'.format(image_path, text_prompt, output_dir)
    os.system(exe_cmd)


if __name__ == '__main__':
    main()
    # mask_path = '/data/yashengsun/Proj/MMEdit/Grounded-Segment-Anything/outputs/mask.jpg'
    # mask = cv2.imread(mask_path)
    # print(np.unique(mask))
    # print(mask.shape)
