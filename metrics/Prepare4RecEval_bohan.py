import os
import sys
import glob
import torch
import torchvision
from torchvision import transforms
from PIL import Image


# Define transformation to be applied to images
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to (224, 224)
    transforms.ToTensor(),           # Convert images to PyTorch tensors
])


def main():
    # image_directory = '/data/yashengsun/Proj/MMEdit/fMRIInstructDiffusion/logs/versatile_dualstream_instruct_pre_post_w_x0_prospect_versatile_dualstream_instruct_pre_post_w_x0_prospect_2024-03-08/visualize/images/val'
    # image_directory = '/data/yashengsun/Proj/MMEditresults_zip/code/diffuer/fmriinstructdiffusion-fusion_explore/recon_path2-981'
    image_directory = '/data/yashengsun/Proj/MMEditresults_zip/code/diffuer/fmriinstructdiffusion-fusion_explore/all_images_path2-981'
    # List all image files in the directory
    # image_files = [f for f in os.listdir(image_directory) if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_files = glob.glob(os.path.join(image_directory, '*.png'))
    # Create a list to store images
    images = []
    # import pdb; pdb.set_trace()

    # Iterate over the image files
    for image_path in image_files:
        # Open the image using PIL (or you can use cv2.imread() from OpenCV)
        image = Image.open(image_path)
        # image_crop = image.crop((0, 512*3, 512, 512*4))
        # Apply transformations
        image = transform(image)
        # Append the image to the list
        images.append(image)

    # Convert list of images to a tensor
    images = torch.stack(images)
    # import pdb; pdb.set_trace()
    # Save the tensor of images as a PyTorch checkpoint
    torch.save(images, 'bohan_gt_images.pt')
    # torch.save(images, 'bohan_pred_images.pt')


if __name__ == '__main__':
    main()
