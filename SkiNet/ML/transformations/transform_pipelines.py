from typing import Literal, cast

import albumentations as A

from SkiNet.ML.configs.transform_configs.augment_config import PhotoAugmentConfig, SpatialAugmentConfig
from SkiNet.ML.configs.transform_configs.crop_config import CropConfig
from SkiNet.ML.configs.transform_configs.transform_config import TransformConfig


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
                                            width=config.size[1],
                                            p=1.0))
    elif config.crop_type == "random_crop":
        transforms_list.append(A.RandomCrop(height=config.size[0],
                                            width=config.size[1],
                                            p=1.0))
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

    if config.elastic_apply:
        transforms_list.append(A.ElasticTransform(alpha=config.elastic_alpha,
                                                  sigma=config.elastic_sigma,
                                                  p=config.elastic_p))

    return transforms_list


def get_photometric_transforms(config: PhotoAugmentConfig) -> list[A.BasicTransform]:
    """
    Returns a list of photometric transformations based on the provided configuration. If no photometric augmentations are applied, an empty list is returned.

    :param config: PhotoAugmentConfig object containing photometric augmentation parameters.
    :return: List of photometric transformations.
    """
    transforms_list: list[A.BasicTransform] = []

    if config.color_jitter_apply:
        transforms_list.append(A.ColorJitter(brightness=config.color_jitter_brightness,  # type: ignore[call-arg]
                                             contrast=config.color_jitter_contrast,
                                             saturation=config.color_jitter_saturation,
                                             hue=config.color_jitter_hue,
                                             p=config.color_jitter_p))

    if config.gaussian_blur_apply:
        transforms_list.append(A.GaussianBlur(sigma_limit=config.gaussian_blur_sigma_limit,  # type: ignore[call-arg]
                                              p=config.gaussian_blur_p))

    if config.gaussian_noise_apply:
        transforms_list.append(A.GaussNoise(std_range=config.gaussian_noise_std_range,
                                            p=config.gaussian_noise_p))

    return transforms_list


def get_postprocess_transforms(config: TransformConfig | None = None) -> list[A.BasicTransform]:
    """
    Returns a list of post-processing transformations to be applied after all augmentations.
    """
    if config is None:
        mode = "image_per_channel"
        mean = std = None
    else:
        mode = config.normalization_mode
        mean = config.normalization_mean
        std = config.normalization_std

    if mode == "standard":
        if mean is None or std is None:
            raise ValueError(
                "normalization_mode='standard' requires normalization_mean and normalization_std "
                "to be set in TRANSFORM_CONFIG. Run compute_dataset_stats.py to obtain them."
            )
        normalize = A.Normalize(normalization="standard", mean=mean, std=std, p=1.0)
    else:
        normalize = A.Normalize(
            normalization=cast(Literal["standard", "image", "image_per_channel", "min_max", "min_max_per_channel"], mode),
            p=1.0,
        )

    return [normalize, A.ToTensorV2(transpose_mask=True)]
