"""
Plot an arbitrary number of images and masks, provided full path to them. Overlap each mask over the corresponding image.
Adapted from https://pytorch.org/vision/stable/auto_examples/others/plot_visualization_utils.html#sphx-glr-auto-examples-others-plot-visualization-utils-py
"""

from typing import List
import torch
from torchvision.utils import draw_segmentation_masks
import numpy as np

from SkiNet.Utils.dev_utils import is_running_in_docker
# Set matplotlib backend based on environment
if is_running_in_docker():
    # to plot in docker
    import matplotlib
    matplotlib.use("Agg")

from IPython.display import display

from matplotlib import pyplot as plt
import torchvision.transforms.functional as F
from torchvision import transforms


import math
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
import torchvision.transforms.functional as F
import numpy as np

def show(imgs: List[torch.Tensor], max_cols=2):
    """
    Show images provided as a list of torch tensors, with automatic wrapping of images
    into multiple rows if needed.

    :param imgs: List of images (torch tensors).
    :param max_cols: Maximum number of columns per row. The rest will be wrapped into the next row.
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    if not imgs:
        raise ValueError("No images provided to show, img cannot be empty")

    # Calculate the number of rows needed based on max_cols
    ncols = min(len(imgs), max_cols)  # Ensure no more than max_cols columns
    nrows = math.ceil(len(imgs) / ncols)  # Wrap images into multiple rows if necessary

    # Create the subplots
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, figsize=(ncols * 5, nrows * 4))

    for i, img in enumerate(imgs):
        img = F.to_pil_image(img)
        
        # Calculate row and column index
        row = i // ncols
        col = i % ncols
        
        axs[row, col].imshow(np.asarray(img))
        axs[row, col].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.tight_layout()

    # Use the right method to show the plot
    if is_running_in_docker():
        display(fig)  # Use this if you want to force inline plotting
        plt.show()
    else:
        plt.show()  # Normal display outside Docker
    
    plt.close(fig)  # Prevents duplicate figures in Jupyter

def plot_masks_over_images(images, masks, alpha=0.3, colors="white", max_cols=2):
    """
    Plot masks over images with transparency given the images and masks as lists of PIL images

    :param images: List of images (e.g. PIL images)
    :param masks: List of masks (e.g. PIL images)
    :param alpha: Transparency of the mask
    :param colors: Color of the mask
    :param max_cols: Maximum number of columns per row. The rest will be wrapped into the next row.
    """
    #
    #transform = transforms.ToTensor()

    transform = transforms.Compose([
        transforms.ToTensor(), # Converts to float32 (0-1)
        transforms.Lambda(lambda x: (x * 255).byte())  # Scale & convert to uint8
    ])

    images_with_masks = []
    for image, mask in zip(images, masks):

        image = transform(image) #sample['image'] is:  torch.Size([1, 128, 128])
        mask = transform(mask) #sample['mask'] is:  torch.Size([1, 128, 128])
        if type(mask) != torch.bool:
            # convert to bool as  draw_segmentation_masks accepts bool tensor as a mask
            mask = mask.to(torch.bool)
        images_with_masks.append(draw_segmentation_masks(image, masks=mask, alpha=alpha, colors = colors))

    show(images_with_masks, max_cols=max_cols)
