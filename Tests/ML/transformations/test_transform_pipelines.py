import albumentations as A
import pytest

from SkiNet.ML.configs.transform_configs.augment_config import PhotoAugmentConfig, SpatialAugmentConfig
from SkiNet.ML.configs.transform_configs.crop_config import CropConfig
from SkiNet.ML.transformations.transform_pipelines import (
    get_crop_transforms,
    get_photometric_transforms,
    get_postprocess_transforms,
    get_spatial_transforms,
)


@pytest.mark.parametrize(
    "config,expected_types",
    [
        (CropConfig(crop_apply=False), []),
        (CropConfig(crop_type="center_crop", size=(128, 96)), ["CenterCrop"]),
        (CropConfig(crop_type="random_crop", size=(128, 96)), ["RandomCrop"]),
        (CropConfig(crop_type="random_resized_crop", size=(128, 96), scale=(0.7, 0.9)), ["RandomResizedCrop"]),
    ],
)
def test_get_crop_transforms_returns_expected_transform_types(
    config: CropConfig,
    expected_types: list[str],
) -> None:
    """
    get_crop_transforms() should return the expected crop transform type for each crop configuration.
    """
    transforms = get_crop_transforms(config)

    assert [type(transform).__name__ for transform in transforms] == expected_types


@pytest.mark.parametrize(
    "config,expected_class,expected_attrs",
    [
        (
            CropConfig(crop_type="center_crop", size=(64, 48)),
            A.CenterCrop,
            {"height": 64, "width": 48},
        ),
        (
            CropConfig(crop_type="random_crop", size=(64, 48)),
            A.RandomCrop,
            {"height": 64, "width": 48},
        ),
        (
            CropConfig(crop_type="random_resized_crop", size=(64, 48), scale=(0.6, 0.8)),
            A.RandomResizedCrop,
            {"size": (64, 48)},
        ),
    ],
)
def test_get_crop_transforms_preserves_requested_crop_size(
    config: CropConfig,
    expected_class: type[A.BasicTransform],
    expected_attrs: dict[str, int | tuple[int, int]],
) -> None:
    """
    get_crop_transforms() should build crop transforms with the configured output size.
    """
    transforms = get_crop_transforms(config)

    assert len(transforms) == 1
    assert isinstance(transforms[0], expected_class)

    for attr_name, expected_value in expected_attrs.items():
        assert getattr(transforms[0], attr_name) == expected_value


@pytest.mark.parametrize(
    "config,expected_types",
    [
        (
            SpatialAugmentConfig(),
            [],
        ),
        (
            SpatialAugmentConfig(square_symmetry_apply=True,
                                 affine_apply=True,
                                 perspective_apply=True),
            ["SquareSymmetry", "Affine", "Perspective"],
        ),
        (
            SpatialAugmentConfig(
                square_symmetry_apply=True,
                affine_apply=True,
                perspective_apply=False,
            ),
            ["SquareSymmetry", "Affine"],
        ),
    ],
)
def test_get_spatial_transforms_returns_expected_transform_types(
    config: SpatialAugmentConfig,
    expected_types: list[str],
) -> None:
    """
    get_spatial_transforms() should include only the enabled spatial augmentation steps in order.
    """
    transforms = get_spatial_transforms(config)

    assert [type(transform).__name__ for transform in transforms] == expected_types


@pytest.mark.parametrize(
    "config,expected_probabilities",
    [
        (
            SpatialAugmentConfig(
                square_symmetry_apply=True,
                square_symmetry_p=0.3,
                perspective_apply=True,
                perspective_p=0.4,
            ),
            {
                "SquareSymmetry": 0.3,
                "Perspective": 0.4,
            },
        ),
    ],
)
def test_get_spatial_transforms_uses_configured_probabilities(
    config: SpatialAugmentConfig,
    expected_probabilities: dict[str, float],
) -> None:
    """
    get_spatial_transforms() should pass configured probabilities through to Albumentations transforms.
    """
    transforms = get_spatial_transforms(config)

    actual_probabilities = {
        type(transform).__name__: transform.p
        for transform in transforms
        if hasattr(transform, "p")
    }

    for transform_name, expected_probability in expected_probabilities.items():
        assert actual_probabilities[transform_name] == expected_probability


@pytest.mark.parametrize(
    "config,expected_types",
    [
        (PhotoAugmentConfig(color_jitter_apply=False), []),
        (PhotoAugmentConfig(color_jitter_apply=True), ["ColorJitter"]),
    ],
)
def test_get_photometric_transforms_returns_expected_transform_types(
    config: PhotoAugmentConfig,
    expected_types: list[str],
) -> None:
    """
    get_photometric_transforms() should include ColorJitter only when it is enabled.
    """
    transforms = get_photometric_transforms(config)

    assert [type(transform).__name__ for transform in transforms] == expected_types


def test_get_photometric_transforms_uses_configured_color_jitter_values() -> None:
    """
    get_photometric_transforms() should pass configured ColorJitter values through unchanged.
    """
    config = PhotoAugmentConfig(
        color_jitter_brightness=0.1,
        color_jitter_contrast=0.2,
        color_jitter_saturation=0.3,
        color_jitter_p=0.4,
    )

    transforms = get_photometric_transforms(config)

    assert len(transforms) == 0  # default is False


@pytest.mark.parametrize(
    "config,expected_types",
    [
        (
            SpatialAugmentConfig(elastic_apply=False),
            [],
        ),
        (
            SpatialAugmentConfig(elastic_apply=True),
            ["ElasticTransform"],
        ),
        (
            SpatialAugmentConfig(square_symmetry_apply=True, affine_apply=True,
                                 perspective_apply=True, elastic_apply=True),
            ["SquareSymmetry", "Affine", "Perspective", "ElasticTransform"],
        ),
    ],
)
def test_get_spatial_transforms_elastic_included_when_enabled(
    config: SpatialAugmentConfig,
    expected_types: list[str],
) -> None:
    """
    get_spatial_transforms() should append ElasticTransform only when elastic_apply is True,
    and place it after Perspective in the transform order.
    """
    transforms = get_spatial_transforms(config)

    assert [type(t).__name__ for t in transforms] == expected_types


def test_get_spatial_transforms_elastic_uses_configured_params() -> None:
    """
    get_spatial_transforms() should pass elastic_alpha, elastic_sigma, and elastic_p
    through to the ElasticTransform instance unchanged.
    """
    config = SpatialAugmentConfig(elastic_apply=True, elastic_alpha=2.0,
                                  elastic_sigma=30.0, elastic_p=0.5)

    transforms = get_spatial_transforms(config)

    assert len(transforms) == 1
    elastic = transforms[0]
    assert isinstance(elastic, A.ElasticTransform)
    assert elastic.alpha == 2.0
    assert elastic.sigma == 30.0
    assert elastic.p == 0.5


@pytest.mark.parametrize(
    "config,expected_types",
    [
        (
            PhotoAugmentConfig(gaussian_blur_apply=False, gaussian_noise_apply=False),
            [],
        ),
        (
            PhotoAugmentConfig(color_jitter_apply=True, gaussian_blur_apply=True,
                               gaussian_noise_apply=True),
            ["ColorJitter", "GaussianBlur", "GaussNoise"],
        ),
        (
            PhotoAugmentConfig(gaussian_blur_apply=True),
            ["GaussianBlur"],
        ),
        (
            PhotoAugmentConfig(gaussian_noise_apply=True),
            ["GaussNoise"],
        ),
    ],
)
def test_get_photometric_transforms_blur_and_noise_included_when_enabled(
    config: PhotoAugmentConfig,
    expected_types: list[str],
) -> None:
    """
    get_photometric_transforms() should include GaussianBlur and GaussNoise only when
    their respective apply flags are True, and in the order: ColorJitter → GaussianBlur → GaussNoise.
    """
    transforms = get_photometric_transforms(config)

    assert [type(t).__name__ for t in transforms] == expected_types


def test_get_photometric_transforms_blur_uses_configured_params() -> None:
    """
    get_photometric_transforms() should pass gaussian_blur_sigma_limit and gaussian_blur_p
    through to the GaussianBlur instance unchanged.
    """
    config = PhotoAugmentConfig(gaussian_blur_apply=True,
                                gaussian_blur_sigma_limit=(1.0, 3.0),
                                gaussian_blur_p=0.4)

    transforms = get_photometric_transforms(config)

    assert len(transforms) == 1
    blur = transforms[0]
    assert isinstance(blur, A.GaussianBlur)
    assert blur.p == 0.4


def test_get_photometric_transforms_noise_uses_configured_params() -> None:
    """
    get_photometric_transforms() should pass gaussian_noise_std_range and gaussian_noise_p
    through to the GaussNoise instance unchanged.
    """
    config = PhotoAugmentConfig(gaussian_noise_apply=True,
                                gaussian_noise_std_range=(0.1, 0.3),
                                gaussian_noise_p=0.6)

    transforms = get_photometric_transforms(config)

    assert len(transforms) == 1
    noise = transforms[0]
    assert isinstance(noise, A.GaussNoise)
    assert noise.p == 0.6


def test_get_postprocess_transforms_returns_normalize_then_tensor() -> None:
    """
    get_postprocess_transforms() should always return exactly [Normalize, ToTensorV2]
    with image_per_channel normalisation and transpose_mask=True.
    """
    transforms: list[A.BasicTransform] = get_postprocess_transforms()

    assert [type(transform).__name__ for transform in transforms] == ["Normalize", "ToTensorV2"]

    normalize = transforms[0]
    to_tensor = transforms[1]

    assert isinstance(normalize, A.Normalize)
    assert normalize.normalization == "image_per_channel"
    assert normalize.p == 1.0

    assert isinstance(to_tensor, A.ToTensorV2)
    assert to_tensor.transpose_mask is True
