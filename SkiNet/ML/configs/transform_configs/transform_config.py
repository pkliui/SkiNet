from typing import Literal

from SkiNet.ML.configs.transform_configs.base_transform_config import BaseTransformConfig
from SkiNet.ML.configs.transform_configs.augment_config import (
    PhotoAugmentConfig,
    SpatialAugmentConfig,
)
from SkiNet.ML.configs.transform_configs.crop_config import CropConfig
from pydantic import Field


class TransformConfig(BaseTransformConfig):
    """
    Main configuration for transformations applied to samples, including cropping and augmentations.
    """
    crop: CropConfig = Field(
        default_factory=CropConfig,
        description=(
            "Cropping configuration for segmentation experiments, including "
            "crop size and method."
        ),
    )
    spatial_augmentation: SpatialAugmentConfig = Field(
        default_factory=SpatialAugmentConfig,
        description=(
            "Spatial augmentation configuration for segmentation experiments, "
            "including random flips, affine transformations, perspective "
            "transformations, and square symmetry."
        ),
    )
    photometric_augmentation: PhotoAugmentConfig = Field(
        default_factory=PhotoAugmentConfig,
        description=(
            "Photometric augmentation configuration for segmentation "
            "experiments, including random brightness/contrast adjustments, "
            "hue/saturation adjustments, and RGB shifts."
        ),
    )
    normalization_mode: Literal["standard", "image_per_channel", "image", "min_max"] = Field(
        default="image_per_channel",
        description=(
            "Albumentations normalization mode. Use 'standard' with dataset-level "
            "mean/std (fastest — fixed constants, no per-image reductions). "
            "'image_per_channel' computes mean/std per sample at runtime and is "
            "~20x slower. 'image' and 'min_max' are intermediate options."
        ),
    )
    normalization_mean: tuple[float, float, float] | None = Field(
        default=None,
        description=(
            "Per-channel mean for 'standard' normalization (R, G, B), values in [0, 1]. "
            "Required when normalization_mode='standard'. Compute with compute_dataset_stats.py."
        ),
    )
    normalization_std: tuple[float, float, float] | None = Field(
        default=None,
        description=(
            "Per-channel std for 'standard' normalization (R, G, B), values in [0, 1]. "
            "Required when normalization_mode='standard'. Compute with compute_dataset_stats.py."
        ),
    )
