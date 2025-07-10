"""
Default configuration for image segmentation.
It uses albumentations library.

References:
Chlap, P., Min, H., Vandenberg, N., Dowling, J., Holloway, L., & Haworth, A. (2021). 
A review of medical image data augmentation techniques for deep learning applications. 
Journal of medical imaging and radiation oncology, 65(5), 545-563.

Basic augmentation techniques:
- geometric (mapping points of the image to new locations)
    - geometric transformations (scaling, translation, rotation, flipping, shear, skew); very common
    - cropping; used when there is class imbalance to even the balance
    - occlusion (removing small patches of the image); used when there is class imbalance to even the balance
- photometric (manipulating the image intensity values)
    - gamma contrast, linear contrast, histogram equalization
    - filtering (convolution to sharpern, blur or smooth)
    - adding noise (Gaussian, salt and pepper, uniform)
    
Deformable augmentation techniques:
    - randomised displacement of pixels
    - spline interpolation (B-splines)
    - deformable iamge registration
    - statistical shape models
"""

from yacs.config import CfgNode

# Define top level configuration nodes for augmentation
config = CfgNode()
config.augmentation = CfgNode()
config.augmentation_off = CfgNode()


#########################
# Geometric transformations
# Random flips as per albumentations.HorizontalFlip and albumentations.VerticalFlip
config.augmentation.horizontal_flip_apply = True
config.augmentation.horizontal_flip = CfgNode()
config.augmentation.horizontal_flip.p = 0.5 # probability of applying the transformation

config.augmentation.vertical_flip_apply = True
config.augmentation.vertical_flip = CfgNode()
config.augmentation.vertical_flip.p = 0.5 # probability of applying the transformation

# Random affine transformations as per albumentations.Affine
config.augmentation.affine_apply = True
config.augmentation.affine = CfgNode()
config.augmentation.affine.rotate = (-90, 90) # degrees
config.augmentation.affine.translate_percent = (0.1, 0.1) # translate in percent
config.augmentation.affine.shear = (0, 30) # degrees


#########################
# Colour augmentation
# Color jitter as per albumentations.ColorJitter
config.augmentation.colorjitter_apply = True
config.augmentation.colorjitter = CfgNode()
config.augmentation.colorjitter.brightness = 0.2 # brightness factor
config.augmentation.colorjitter.contrast = 0.2 # contrast factor
config.augmentation.colorjitter.saturation = 0.2 # saturation factor
config.augmentation.colorjitter.p = 0.5 # probability of applying the transformation

#########################
# Cropping and resizing
# Center crop as per albumentations.CenterCrop
config.augmentation.center_crop_apply = True
config.augmentation.center_crop = CfgNode()
config.augmentation.center_crop.height = 500
config.augmentation.center_crop.width = 500


def get_default_config() -> CfgNode:
    """
    Return a clone of the default configuration as a yacs CfgNode object with default values.

    This is useful for creating a new configuration object with the default values, 
    without a risk of modifying the original configuration.
    This is for the "local variable" use pattern.
    """
    return config.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = config  # users can `from transformations_config import cfg`