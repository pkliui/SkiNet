"""
This is a default configuration for image segmentation.
One shall use pytorch.torchvision.v2 for this configuration.

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


####################################################################################################
# Transformations applied when augmentation_required=True in make_transform_from_config

#########################
# Geometric transformations
# Random affine transformations as per torchvision.transforms.v2.RandomAffine
config.augmentation.random_affine_apply = True
config.augmentation.random_affine = CfgNode()
config.augmentation.random_affine.degrees = 30
config.augmentation.random_affine.translate = (0.05, 0.05)
config.augmentation.random_affine.scale = (0.1, 0.3)
config.augmentation.random_affine.shear = 30

# Random rotation as per torchvision.transforms.v2.RandomRotation
config.augmentation.random_rotation_apply = True
config.augmentation.random_rotation = CfgNode()
config.augmentation.random_rotation.degrees = 30

# Random flips as per torchvision.transforms.v2.RandomHorizontalFlip and RandomVerticalFlip
config.augmentation.random_horizontal_flip_apply = True
config.augmentation.random_vertical_flip_apply = True
config.augmentation.random_horizontal_flip = CfgNode()
config.augmentation.random_horizontal_flip.p = 0.5 # probability of applying the transformation
config.augmentation.random_vertical_flip = CfgNode()
config.augmentation.random_vertical_flip.p = 0.5 # probability of applying the transformation

#########################
# Deformable augmentation
# Elastic transforms as per torchvision.transforms.v2.ElasticTransform
config.augmentation.elastic_transforms_apply = True
config.augmentation.elastic_transforms = CfgNode()
config.augmentation.elastic_transforms.alpha = 50.0
config.augmentation.elastic_transforms.sigma = 5.0

# Random perspective as per torchvision.transforms.v2.RandomPerspective
config.augmentation.random_perspective_apply = True
config.augmentation.random_perspective = CfgNode()
config.augmentation.random_perspective.distortion_scale = 0.5
config.augmentation.random_perspective.p = 0.5 # probability of applying the transformation

#########################
# Photometric transformations
# Random equalization of the historgram as per torchvision.transforms.v2.RandomEqualize
config.augmentation.random_equalize_apply = True
config.augmentation.random_equalize = CfgNode()
config.augmentation.random_equalize.p = 0.5 # probability of applying the transformation

# Brightness and contrast as per torchvision.transforms.v2.ColorJitter
config.augmentation.random_colorjitter_apply = True
config.augmentation.random_colorjitter = CfgNode()
config.augmentation.random_colorjitter.brightness = 0.5 # brightness factor
config.augmentation.random_colorjitter.contrast = 0.5 # contrast factor
config.augmentation.random_colorjitter.saturation = 0.5 # saturation factor
config.augmentation.random_colorjitter.hue = 0.5 # hue factor


#########################
# Resize as per torchvision.transforms.v2.Resize
config.augmentation.resize_apply = False
config.augmentation.resize = CfgNode()
config.augmentation.resize.size = (500, 500)

# Centre crop  as per torchvision.transforms.v2.CenterCrop
config.augmentation.crop_apply = True
config.augmentation.center_crop = CfgNode()
config.augmentation.center_crop.size = (500, 500)



####################################################################################################
# Transformations applied when augmentation_required=False in make_transform_from_config

#########################
# Resize as per torchvision.transforms.v2.Resize
config.augmentation_off.resize_apply = False
config.augmentation_off.resize = CfgNode()
config.augmentation_off.resize.size = (500, 500)

# Centre crop  as per torchvision.transforms.v2.CenterCrop
config.augmentation_off.crop_apply = True
config.augmentation_off.center_crop = CfgNode()
config.augmentation_off.center_crop.size = (500, 500)



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