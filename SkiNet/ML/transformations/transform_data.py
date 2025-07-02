import logging
import random
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
import torchvision.transforms.v2 as T
from torchvision.tv_tensors import Image as TVImage
from yacs.config import CfgNode

ImageData = Union[torch.Tensor, 
                  PIL.Image.Image,
                  TVImage, 
                  Tuple[torch.Tensor, torch.Tensor], 
                  Tuple[TVImage, TVImage], 
                  Tuple[PIL.Image.Image, PIL.Image.Image]]


class TransformData:
    def __init__(self, 
                 pipeline: T.Compose):
        """
        TransformData class to apply a series of transformations to images and masks.
        
        :param pipeline: A torchvision.transforms.Compose object containing the transformation pipeline.
        """
        self.pipeline = pipeline

    def ensure_tv_image(self, x: ImageData) -> TVImage:
        """
        Ensure the input is a torchvision Image
        :param x: The input image, which can be a torch.Tensor, TVImage, or PIL.Image.Image or a tuple of these types.
        :return: A torchvision Image object.
        """
        if isinstance(x, TVImage):
            return x
        return T.ToImage()(x)

    def apply_transforms(self, image: ImageData) -> TVImage:
        """
        Apply the transformation pipeline to the input

        According to the torch documentation, v2.transforms support arbitrary input structures, such as single image, a tuple or a dictionary. 
        The same structure will be returned as output. Pure torch.Tensor objects are treated as images. 
        However, if the input is an Image, Video, or PIL.Image.Image instance, all other pure tensors are passed-through (not transformed).
        If there is no Image or Video instance, only the first pure torch.Tensor will be transformed as image or video, while all others will be passed-through. Here “first” means “first in a depth-wise traversal”.
        https://docs.pytorch.org/vision/main/auto_examples/transforms/plot_transforms_getting_started.html#sphx-glr-auto-examples-transforms-plot-transforms-getting-started-py

        Hence, if the input is a tuple of torch.Image instances, all images will be transformed, 
        but if the input is a tuple of torch.Tensor, only the first tensor will be transformed.
        
        :param image: Input provided as a torch.Tensor, TVImage, PIL.Image.Image, or a tuple of these types.
        :return: The transformed input, where all images in the input are transformed according to the pipeline.
        """

        if isinstance(image, tuple):
            # Recursively ensure that each element in the tuple is of type torchvision.tv_tensors.Image
            # typically this should have been done in the dataset whilst reading out data
            # double check here in case this has not been done in the dataset
            image = tuple(self.ensure_tv_image(im) for im in image)
        else:
            image = self.ensure_tv_image(image)

        # at this point the input to the pipeline is TVImage
        return self.pipeline(image)

    def __call__(self, image: Union[torch.Tensor, TVImage]) -> torch.Tensor:
        return self.apply_transforms(image)


def make_transform_from_config(config: CfgNode, augmentation_required: bool):
    """
    Create transforms for augmentation and resizing using torchvision.transforms.v2.

    :param config: Configuration object containing augmentation parameters
    :param augmentation_required: Boolean flag indicating whether augmentation is required
        If False, only resize and crop are applied.
    :return: A torchvision transform.Compose object
    """
    transforms_list = []

    if augmentation_required:
        #########################
        # Geometric transformations
        # Random affine transformations as per torchvision.transforms.v2.RandomAffine
        if config.augmentation.random_affine_apply:
            transforms_list.append(
                T.RandomAffine(
                    degrees=config.augmentation.random_affine.degrees,
                    translate=config.augmentation.random_affine.translate,
                    scale=config.augmentation.random_affine.scale,
                    shear=config.augmentation.random_affine.shear
                )
            )
        # Random rotation as per torchvision.transforms.v2.RandomRotation
        if config.augmentation.random_rotation_apply:
            transforms_list.append(
                T.RandomRotation(
                    degrees=config.augmentation.random_rotation.degrees
                )
            )
        # Random flips as per torchvision.transforms.v2.RandomHorizontalFlip and RandomVerticalFlip
        if config.augmentation.random_horizontal_flip_apply:
            transforms_list.append(
                T.RandomHorizontalFlip(p=config.augmentation.random_horizontal_flip.p)
            )
        if config.augmentation.random_vertical_flip_apply:
            transforms_list.append(
                T.RandomVerticalFlip(p=config.augmentation.random_vertical_flip.p)
            )

        #########################
        # Deformable augmentation
        # Elastic transforms as per torchvision.transforms.v2.ElasticTransform
        if config.augmentation.elastic_transforms_apply:
            transforms_list.append(
                T.ElasticTransform(
                    alpha=config.augmentation.elastic_transforms.alpha,
                    sigma=config.augmentation.elastic_transforms.sigma
                )
            )
        # Random perspective as per torchvision.transforms.v2.RandomPerspective
        if config.augmentation.random_perspective_apply:
            transforms_list.append(
                T.RandomPerspective(
                    distortion_scale=config.augmentation.random_perspective.distortion_scale,
                    p=config.augmentation.random_perspective.p
                )
            )

        #########################
        # Photometric transformations
        # Random equalization of the historgram as per torchvision.transforms.v2.RandomEqualize
        if config.augmentation.random_equalize_apply:
            transforms_list.append(
                T.RandomEqualize(p=config.augmentation.random_equalize.p)
            )
        # Brightness and contrast as per torchvision.transforms.v2.ColorJitter
        if config.augmentation.random_colorjitter_apply:
            transforms_list.append(
                T.ColorJitter(
                    brightness=config.augmentation.random_colorjitter.brightness,
                    contrast=config.augmentation.random_colorjitter.contrast,
                    saturation=config.augmentation.random_colorjitter.saturation,
                    hue=config.augmentation.random_colorjitter.hue
                )
            )
        
        #########################
        # Resize  as per torchvision.transforms.v2.Resize
        if config.augmentation.resize_apply:
            transforms_list.append(
                T.Resize(size=config.augmentation.resize.size)
                )
        # Centre crop  as per torchvision.transforms.v2.CenterCrop
        if config.augmentation.crop_apply:
            transforms_list.append(
                T.CenterCrop(size=config.augmentation.center_crop.size)
                )
    else:
        # Resize  as per torchvision.transforms.v2.Resize
        if config.augmentation_off.resize_apply:
            transforms_list.append(
                T.Resize(size=config.augmentation_off.resize.size)
                )
        # Centre crop  as per torchvision.transforms.v2.CenterCrop
        if config.augmentation_off.crop_apply:
            transforms_list.append(
                T.CenterCrop(size=config.augmentation_off.center_crop.size)
                )
            
    transforms_list.append(T.ToDtype(torch.float32, scale=True))
    pipeline = TransformData(T.Compose(transforms_list))

    return pipeline
