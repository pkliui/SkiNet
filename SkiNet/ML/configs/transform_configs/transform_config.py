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
