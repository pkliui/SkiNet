import albumentations as A

from SkiNet.ML.configs.transform_configs.augment_config import PhotoAugmentConfig, SpatialAugmentConfig
from SkiNet.ML.configs.transform_configs.crop_config import CropConfig


def get_crop_transforms(config: CropConfig) -> list[A.BasicTransform]:
    """
    Returns a list of cropping transformations based on the provided configuration. If cropping is not applied, an empty list is returned.

    :param config: CropConfig object containing cropping parameters.
    :return: List of cropping transformations.
    """
    transforms_list: list[A.BasicTransform] = []
    if not config.crop_apply:
        return transforms_list

    if config.crop_type == "center_crop":
        transforms_list.append(A.CenterCrop(height=config.size[0],
                                            width=config.size[1]))
    elif config.crop_type == "random_crop":
        transforms_list.append(A.RandomCrop(height=config.size[0],
                                            width=config.size[1]))
    elif config.crop_type == "random_resized_crop":
        transforms_list.append(A.RandomResizedCrop(size=(config.size[0],
                                                         config.size[1]),
                                                   scale=config.scale,
                                                   p=1.0))
    return transforms_list


def get_spatial_transforms(config: SpatialAugmentConfig) -> list[A.BasicTransform]:
    """
    Returns a list of spatial transformations based on the provided configuration. If no spatial augmentations are applied, an empty list is returned.

    :param config: SpatialAugmentConfig object containing spatial augmentation parameters.
    :return: List of spatial transformations.
    """
    transforms_list: list[A.BasicTransform] = []

    if config.horizontal_flip_apply:
        transforms_list.append(A.HorizontalFlip(p=config.horizontal_flip_p))

    if config.vertical_flip_apply:
        transforms_list.append(A.VerticalFlip(p=config.vertical_flip_p))

    if config.square_symmetry_apply:
        transforms_list.append(A.SquareSymmetry(p=config.square_symmetry_p))

    if config.affine_apply:
        transforms_list.append(A.Affine(scale=config.affine_scale,
                                        translate_percent=config.affine_translate_percent,
                                        rotate=config.affine_rotate,
                                        shear=config.affine_shear))

    if config.perspective_apply:
        transforms_list.append(A.Perspective(scale=config.perspective_scale,
                                             p=config.perspective_p))

    return transforms_list


def get_photometric_transforms(config: PhotoAugmentConfig) -> list[A.BasicTransform]:
    """
    Returns a list of photometric transformations based on the provided configuration. If no photometric augmentations are applied, an empty list is returned.

    :param config: PhotoAugmentConfig object containing photometric augmentation parameters.
    :return: List of photometric transformations.
    """
    transforms_list: list[A.BasicTransform] = []

    if config.color_jitter_apply:
        transforms_list.append(A.ColorJitter(brightness=config.color_jitter_brightness,
                                             contrast=config.color_jitter_contrast,
                                             saturation=config.color_jitter_saturation,
                                             hue=config.color_jitter_hue,
                                             p=config.color_jitter_p))

    return transforms_list


def get_postprocess_transforms() -> list[A.BasicTransform]:
    """
    Returns a list of post-processing transformations to be applied after all augmentations.
    This typically includes normalization and conversion to tensor format.
    """
    return [A.Normalize(), A.ToTensorV2(transpose_mask=True)]
