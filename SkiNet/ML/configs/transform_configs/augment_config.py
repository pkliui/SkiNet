from typing import Tuple

from pydantic import BaseModel, Field


class SpatialAugmentConfig(BaseModel):
    """
    Default configuration for spatial image augmenation in segmentation experiments using Albumentations library.
    Note that spatial augmentations are geometric transformations that manipulate the spatial arrangement of pixels in an image.

    References:
    Chlap, P., Min, H., Vandenberg, N., Dowling, J., Holloway, L., & Haworth, A. (2021).
    A review of medical image data augmentation techniques for deep learning applications.
    Journal of medical imaging and radiation oncology, 65(5), 545-563

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
        - deformable image registration
        - statistical shape models
    """
    # Geometric transformations
    # albumentations.HorizontalFlip and albumentations.VerticalFlip
    horizontal_flip_apply: bool = Field(
        default=True, description="Apply horizontal flip.")
    horizontal_flip_p: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Probability of horizontal flip.")
    #
    vertical_flip_apply: bool = Field(
        default=True, description="Apply vertical flip.")
    vertical_flip_p: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Probability of vertical flip.")

    # Square symmetry (Vertical Flip & 90/180/270 Rotations)
    square_symmetry_apply: bool = Field(
        default=True, description="Apply square symmetry transformations.")
    square_symmetry_p: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Probability of square symmetry transformations.")

    # Affine transformations
    # albumentations.Affine
    affine_apply: bool = Field(
        default=True, description="Apply affine transformations.")
    affine_scale: Tuple[float, float] = Field(default=(
        0.9, 1.1), description="Scaling (zoom) factor for affine transformation.")
    affine_translate_percent: dict[str, float | Tuple[float, float]] = Field(default={"x": (
        0.0, 0.2), "y": (0.0, 0.2)}, description="Translation as a fraction of the image size.")
    affine_rotate: Tuple[float, float] = Field(
        default=(-15, 15), description="Rotation angle in degrees.")
    affine_shear: dict[str, float | Tuple[float, float]] = Field(
        default={"x": (0, 20), "y": (0, 20)}, description="Shear angle in degrees.")

    # Perspective transformations
    # albumentations.Perspective
    perspective_apply: bool = Field(
        default=True, description="Apply perspective transformations.")
    perspective_scale: Tuple[float, float] = Field(default=(
        0.9, 1.1), description="Scaling factor for perspective transformation.")
    perspective_p: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Probability of applying perspective transformation.")


class PhotoAugmentConfig(BaseModel):
    """
    Default configuration for photometric image augmenation in segmentation experiments using Albumentations library.
    Note that photometric augmentations are transformations that manipulate the
    intensity values of pixels in an image without changing their spatial
    arrangement.

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
        - deformable image registration
        - statistical shape models
    """

    # albumentations.ColorJitter
    color_jitter_apply: bool = Field(
        default=True, description="Apply color jitter.")
    color_jitter_brightness: float = Field(
        default=0.2, description="Brightness adjustment factor.")
    color_jitter_contrast: float = Field(
        default=0.2, description="Contrast adjustment factor.")
    color_jitter_saturation: float = Field(
        default=0.2, description="Saturation adjustment factor.")
    color_jitter_p: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Probability of applying color jitter.")
