import pytest

from SkiNet.ML.configs.transform_configs.augment_config import PhotoAugmentConfig, SpatialAugmentConfig
from SkiNet.ML.configs.transform_configs.crop_config import CropConfig
from SkiNet.ML.configs.transform_configs.transform_config import TransformConfig


@pytest.mark.parametrize(
    "config,expected_apply,expected_type,expected_size,expected_scale",
    [
        (CropConfig(), True, "random_resized_crop", (500, 500), (0.8, 1.0)),
        (
            CropConfig(crop_apply=False, crop_type="center_crop", size=(64, 48), scale=(0.6, 0.9)),
            False,
            "center_crop",
            (64, 48),
            (0.6, 0.9),
        ),
    ],
)
def test_crop_config_uses_defaults_and_accepts_overrides(
    config: CropConfig,
    expected_apply: bool,
    expected_type: str,
    expected_size: tuple[int, int],
    expected_scale: tuple[float, float],
) -> None:
    """
    CropConfig should expose documented defaults and replace them with explicit override values.
    """
    assert config.crop_apply is expected_apply
    assert config.crop_type == expected_type
    assert config.size == expected_size
    assert config.scale == expected_scale


@pytest.mark.parametrize(
    "config,expected_flip_p,expected_affine_scale,expected_perspective_p,expected_shear_x",
    [
        (SpatialAugmentConfig(), 0.5, (0.9, 1.1), 0.5, (0, 20)),
        (
            SpatialAugmentConfig(
                horizontal_flip_p=0.2,
                affine_scale=(0.8, 1.2),
                perspective_p=0.9,
                affine_shear={"x": (1, 3), "y": (4, 6)},
            ),
            0.2,
            (0.8, 1.2),
            0.9,
            (1, 3),
        ),
    ],
)
def test_spatial_augment_config_uses_defaults_and_accepts_overrides(
    config: SpatialAugmentConfig,
    expected_flip_p: float,
    expected_affine_scale: tuple[float, float],
    expected_perspective_p: float,
    expected_shear_x: tuple[int, int],
) -> None:
    """
    SpatialAugmentConfig should use model defaults unless explicit augmentation values are provided.
    """
    assert config.horizontal_flip_p == expected_flip_p
    assert config.affine_scale == expected_affine_scale
    assert config.perspective_p == expected_perspective_p
    assert config.affine_shear["x"] == expected_shear_x


@pytest.mark.parametrize(
    "config,expected_apply,expected_brightness,expected_contrast,expected_saturation,expected_p",
    [
        (PhotoAugmentConfig(), True, 0.2, 0.2, 0.2, 0.5),
        (
            PhotoAugmentConfig(
                color_jitter_apply=False,
                color_jitter_brightness=0.1,
                color_jitter_contrast=0.3,
                color_jitter_saturation=0.4,
                color_jitter_p=0.8,
            ),
            False,
            0.1,
            0.3,
            0.4,
            0.8,
        ),
    ],
)
def test_photo_augment_config_uses_defaults_and_accepts_overrides(
    config: PhotoAugmentConfig,
    expected_apply: bool,
    expected_brightness: float,
    expected_contrast: float,
    expected_saturation: float,
    expected_p: float,
) -> None:
    """
    PhotoAugmentConfig should expose default jitter settings and allow explicit override values.
    """
    assert config.color_jitter_apply is expected_apply
    assert config.color_jitter_brightness == expected_brightness
    assert config.color_jitter_contrast == expected_contrast
    assert config.color_jitter_saturation == expected_saturation
    assert config.color_jitter_p == expected_p


def test_transform_config_builds_nested_defaults() -> None:
    """
    TransformConfig should create default nested crop, spatial, and photometric configs when no overrides are provided.
    """
    config = TransformConfig()

    assert isinstance(config.crop, CropConfig)
    assert isinstance(config.spatial_augmentation, SpatialAugmentConfig)
    assert isinstance(config.photometric_augmentation, PhotoAugmentConfig)
    assert config.crop.crop_type == "random_resized_crop"
    assert config.spatial_augmentation.horizontal_flip_p == 0.5
    assert config.photometric_augmentation.color_jitter_p == 0.5


def test_transform_config_applies_nested_overrides() -> None:
    """
    TransformConfig should apply explicit nested overrides to the appropriate child config models.
    """
    config = TransformConfig(
        crop=CropConfig(crop_type="center_crop", size=(64, 48)),
        spatial_augmentation=SpatialAugmentConfig(horizontal_flip_p=0.2, perspective_apply=False),
        photometric_augmentation=PhotoAugmentConfig(color_jitter_apply=False, color_jitter_p=0.8),
    )

    assert config.crop.crop_type == "center_crop"
    assert config.crop.size == (64, 48)
    assert config.spatial_augmentation.horizontal_flip_p == 0.2
    assert config.spatial_augmentation.perspective_apply is False
    assert config.photometric_augmentation.color_jitter_apply is False
    assert config.photometric_augmentation.color_jitter_p == 0.8
