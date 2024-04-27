from segment_anything import SamPredictor, sam_model_registry
from PIL import Image


def main():
    model_type = 'vit_h'
    ckpt_path = '/data/yashengsun/Proj/Diffusion/sam_vit_h_4b8939.pth'
    sam = sam_model_registry[model_type](checkpoint=ckpt_path)
    predictor = SamPredictor(sam)
    your_image_path = '../../Diffusion/InstructDiffusion/teaser_samples/all_iter-999999_ep-999999_bidx-000234-020064.png'
    your_image = Image.open(your_image_path)
    predictor.set_image(your_image)
    masks, _, _ = predictor.predict('cat.')
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
