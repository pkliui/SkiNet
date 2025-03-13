"""
Plot an arbitrary number of images and masks, provided full path to them. Overlap each mask over the corresponding image.
Adapted from https://pytorch.org/vision/stable/auto_examples/others/plot_visualization_utils.html#sphx-glr-auto-examples-others-plot-visualization-utils-py
"""

from typing import List
import torch
from torchvision.utils import draw_segmentation_masks
from torchvision.io import decode_image, read_file
import numpy as np
from matplotlib import pyplot as plt
import torchvision.transforms.functional as F
from torchvision import transforms


def show(imgs: List[torch.Tensor]):
    """
    Show images provided as a list of torch tensors
    """
    if not isinstance(imgs, list):
        imgs = [imgs]

    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.show()

def plot_masks_over_images(images, masks, alpha=0.3, colors="white"):
    """
    Plot masks over images with transparency given the images and masks as lists of PIL images

    :param images: List of images (e.g. PIL images)
    :param masks: List of masks (e.g. PIL images)
    :param alpha: Transparency of the mask
    :param colors: Color of the mask
    """
    #
    transform = transforms.ToTensor()

    images_with_masks = []
    for image, mask in zip(images, masks):

        image = transform(image) #sample['image'] is:  torch.Size([1, 128, 128])
        mask = transform(mask) #sample['mask'] is:  torch.Size([1, 128, 128])
        if type(mask) != torch.bool:
            # convert to bool as  draw_segmentation_masks accepts bool tensor as a mask
            mask = mask.to(torch.bool)
        images_with_masks.append(draw_segmentation_masks(image, masks=mask, alpha=alpha, colors = colors))

    show(images_with_masks)
