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

    - geometric: geometric transformations (scaling, translation, rotation, flipping, shear, skew); cropping; occlusion
    - photometric: gamma contrast, linear contrast, histogram equalization; filtering; adding noise (Gaussian, salt and pepper, uniform)

    Deformable augmentation techniques:

    - randomised displacement of pixels
    - spline interpolation (B-splines)
    - deformable image registration
    - statistical shape models
    """
    # Geometric transformations

    # Square symmetry (Horizontal, Vertical Flip & 90/180/270 Rotations)
    square_symmetry_apply: bool = Field(
        default=False, description="Apply square symmetry transformations.")
    square_symmetry_p: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Probability of square symmetry transformations.")

    # Affine transformations
    # albumentations.Affine
    affine_apply: bool = Field(
        default=False, description="Apply affine transformations.")
    affine_scale: Tuple[float, float] = Field(default=(
        0.8, 1.0), description="Scaling (zoom) factor for affine transformation.")
    affine_translate_percent: dict[str, float | Tuple[float, float]] = Field(
        default_factory=lambda: dict[str, float | Tuple[float, float]]({"x": (-0.05, 0.05), "y": (-0.05, 0.05)}),
        description="Translation as a fraction of the image size.",
    )
    affine_rotate: Tuple[float, float] = Field(
        default=(-45, 45), description="Rotation angle in degrees.")
    affine_shear: dict[str, float | Tuple[float, float]] = Field(
        default_factory=lambda: dict[str, float | Tuple[float, float]]({"x": (-15.0, 15.0), "y": (-15.0, 15.0)}),
        description="Shear angle in degrees.",
    )

    # Perspective transformations
    # albumentations.Perspective
    perspective_apply: bool = Field(
        default=False, description="Apply perspective transformations.")
    perspective_scale: Tuple[float, float] = Field(default=(
        0.05, 0.1), description="Scaling factor for perspective transformation.")
    perspective_p: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Probability of applying perspective transformation.")

    # Elastic deformation
    # albumentations.ElasticTransform
    elastic_apply: bool = Field(
        default=False, description="Apply elastic deformation.")
    elastic_alpha: float = Field(
        default=120.0, description="Scaling factor controlling displacement magnitude.")
    elastic_sigma: float = Field(
        default=10.0, description="Gaussian smoothing factor; larger values produce smoother deformation.")
    elastic_p: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Probability of applying elastic deformation.")


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

    Photometric augmentation techniques:
    - gamma contrast, linear contrast, histogram equalization
    - filtering (convolution to sharpen, blur or smooth)
    - adding noise (Gaussian, salt and pepper, uniform)

    Implemented transforms:
    - ColorJitter (albumentations.ColorJitter)
    - GaussianBlur (albumentations.GaussianBlur)
    - GaussNoise (albumentations.GaussNoise)
    """

    # albumentations.ColorJitter
    color_jitter_apply: bool = Field(
        default=False, description="Apply color jitter.")
    color_jitter_brightness: float = Field(
        default=0.2, description="Brightness adjustment factor.")
    color_jitter_contrast: float = Field(
        default=0.2, description="Contrast adjustment factor.")
    color_jitter_saturation: float = Field(
        default=0.0, description="Saturation adjustment factor.")
    color_jitter_hue: float = Field(
        default=0.0, description="Hue adjustment factor.")
    color_jitter_p: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Probability of applying color jitter.")

    # albumentations.GaussianBlur
    gaussian_blur_apply: bool = Field(
        default=False, description="Apply Gaussian blur.")
    gaussian_blur_sigma_limit: Tuple[float, float] = Field(
        default=(0.5, 2.0), description="Sigma range for Gaussian blur kernel.")
    gaussian_blur_p: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Probability of applying Gaussian blur.")

    # albumentations.GaussNoise
    gaussian_noise_apply: bool = Field(
        default=False, description="Apply Gaussian noise.")
    gaussian_noise_std_range: Tuple[float, float] = Field(
        default=(0.05, 0.15), description="Std range as a fraction of max pixel value.")
    gaussian_noise_p: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Probability of applying Gaussian noise.")
