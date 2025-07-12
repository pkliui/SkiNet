from typing import Any, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
import torchvision.transforms.v2 as T
import albumentations as A
from torchvision.tv_tensors import Image as TVImage
from yacs.config import CfgNode

from SkiNet.Utils.image_utils import ensure_np_image

# possible input image data types that will be converted to np.ndarray as per compatibility with albumentaions
ImageData = Union[np.ndarray,
                  torch.Tensor, 
                  PIL.Image.Image]


class TransformData:
    def __init__(self, 
                 pipeline: A.Compose):
        """
        TransformData class to apply a series of transformations to images and masks.
        
        :param pipeline: An albumentations.Compose object containing the transformation pipeline.
        """
        self.pipeline = pipeline

    def apply_transforms(self, image: ImageData, mask: Optional[ImageData] = None) -> dict:
        """
        Apply the transformation pipeline to the input, assuming the use of Albumentations library

        :param image: Input image, must be of shape (H, W, C) or (H, W) if C==1.
        :param mask: Optional input mask, must be of shape (H, W, C) or (H, W) if C==1

        :return: The transformed input returned as a dictionary with leys according to the inputs.
            For example, If mask is provided, the dit will have key 'mask'.
            Otherwise, it will have only key 'image'
            As per augmentations library internals, the output dimensions are (C, H, W) unless C==1, then (H, W)
        """

        image = ensure_np_image(image)
        if mask is not None:
            mask = ensure_np_image(mask)
            return self.pipeline(image = image, mask = mask)
        else:
            return self.pipeline(image = image)

    def __call__(self, image: ImageData, mask: Optional[ImageData] = None) -> dict:
        """
        Return transformed image and mask as the dictionary's entries with keys 'image' and 'mask',
        as per Albumentations library conventions.
        If mask is not provided, only the image is transformed.
        """
        if mask is not None:
            transformed = self.apply_transforms(image=image, mask=mask)
            return {'image': transformed['image'], 'mask': transformed['mask']}
        else:
            transformed = self.apply_transforms(image=image)
            return {'image': transformed['image']}

def make_transform_from_config(config: CfgNode, 
                               augmentation_required: bool, 
                               seed_value: Optional[int] = None) -> TransformData:
    """
    Create transforms for augmentation and resizing using Albumentations library

    :param config: Configuration object is a CfgNode object set up as in SkiNet/ML/configs/transformations_config.py.
        Typically the default configuration in there should be overriden by a YAML configuration file for a specific experiment.
    :param augmentation_required: Boolean flag indicating whether augmentation is required
        If False, only transforms under augmentations_off are applied.
    :seed_value: Optional seed value for reproducibility of the transformations. 
        In Albumentations, it is provided as a seed to the Compose object. Note that Albumentations uses its own internal random state 
        that is completely independent from global random seeds. 

    :return: An instance of TransformData
    """
    transforms_list = []

    #########################

    if augmentation_required:
        #########################
        # Geometric transformations

        # Random flips as per albumentations.HorizontalFlip and albumentations.VerticalFlip
        if config.augmentation.horizontal_flip_apply:
            transforms_list.append(
                A.HorizontalFlip(p=config.augmentation.horizontal_flip.p)
            )
        if config.augmentation.vertical_flip_apply:
            transforms_list.append(
                A.VerticalFlip(p=config.augmentation.vertical_flip.p)
            )

        # Random affine transformations as per albumentations.Affine
        if config.augmentation.affine_apply:
            transforms_list.append(
                A.Affine(
                    rotate=config.augmentation.affine.rotate,
                    translate_percent=config.augmentation.affine.translate_percent,
                    shear=config.augmentation.affine.shear
                )
            )

        ##########################
        # Colour transforms
        # Random perspective as per albumentations.ColorJitter
        if config.augmentation.colorjitter_apply:
            transforms_list.append(
                A.ColorJitter(
                    brightness=config.augmentation.colorjitter.brightness,
                    contrast=config.augmentation.colorjitter.contrast,
                    saturation=config.augmentation.colorjitter.saturation,
                    p=config.augmentation.colorjitter.p
                )
            )

        ################################################
        # Center crop as per albumentations.CenterCrop
        if config.augmentation.center_crop_apply:
            transforms_list.append(
                A.CenterCrop(height=config.augmentation.center_crop.height,
                             width=config.augmentation.center_crop.width)
            )

            
    transforms_list.append(A.ToTensorV2())  # Convert images to torch.Tensor
    if seed_value is not None:
        pipeline = TransformData(A.Compose(transforms_list, seed=seed_value))
    else:
        pipeline = TransformData(A.Compose(transforms_list))
    
    return pipeline