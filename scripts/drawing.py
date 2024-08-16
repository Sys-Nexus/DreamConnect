import cv2
import numpy as np
import glob
import os


def dilate_image(image_path, kernel_size=(3, 3), iterations=10):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)[:,:,:3]
    print(image.shape)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # Create a kernel (structuring element)
    kernel = np.ones(kernel_size, np.uint8)

    # Apply dilation
    dilated_image = cv2.dilate(image, kernel, iterations=iterations)
    # dilated_image = image

    # Save the dilated image to a file
    dilated_image_path = os.path.basename(image_path)
    cv2.imwrite(dilated_image_path, dilated_image)
    print(f"Dilated image saved to {dilated_image_path}")


def main():
    root = 'pics/'
    paths = glob.glob(os.path.join(root, '*.png'))
    print(paths)

    # for path in paths:
    # dilate_image(paths[0])
    # dilate_image(paths[1], kernel_size=(3, 3), iterations=30)
    create_brain_shape_background()


if __name__ == '__main__':
    main()
