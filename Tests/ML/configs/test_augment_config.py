from SkiNet.ML.configs.transform_configs.augment_config import PhotoAugmentConfig, SpatialAugmentConfig
from pydantic import ValidationError
import pytest


def test_spatialaugmentconfig_defaults() -> None:
    cfg = SpatialAugmentConfig()
    assert cfg.square_symmetry_p == 0.5
    assert cfg.affine_scale == (0.8, 1.0)
    assert cfg.affine_rotate == (-45, 45)
    assert cfg.affine_shear["x"] == (-15, 15)
    assert cfg.perspective_p == 0.2


def test_spatialaugmentconfig_overrides() -> None:
    cfg = SpatialAugmentConfig(
        affine_scale=(0.8, 1.2),
        perspective_p=0.9,
        affine_shear={"x": (1, 3), "y": (4, 6)},
    )
    assert cfg.affine_scale == (0.8, 1.2)
    assert cfg.perspective_p == 0.9
    assert cfg.affine_shear["x"] == (1, 3)


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_spatialaugmentconfig_probability_out_of_range_raises(p: float) -> None:
    with pytest.raises(ValidationError):
        SpatialAugmentConfig(square_symmetry_p=p)


def test_spatialaugmentconfig_elastic_defaults() -> None:
    cfg = SpatialAugmentConfig()
    assert cfg.elastic_apply is False
    assert cfg.elastic_alpha == 120.0
    assert cfg.elastic_sigma == 10.0
    assert cfg.elastic_p == 0.3


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_spatialaugmentconfig_elastic_probability_out_of_range_raises(p: float) -> None:
    with pytest.raises(ValidationError):
        SpatialAugmentConfig(elastic_p=p)


def test_photoaugmentconfig_blur_and_noise_defaults() -> None:
    cfg = PhotoAugmentConfig()
    assert cfg.gaussian_blur_apply is False
    assert cfg.gaussian_blur_sigma_limit == (0.5, 2.0)
    assert cfg.gaussian_blur_p == 0.2
    assert cfg.gaussian_noise_apply is False
    assert cfg.gaussian_noise_std_range == (0.05, 0.15)
    assert cfg.gaussian_noise_p == 0.2


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_photoaugmentconfig_blur_probability_out_of_range_raises(p: float) -> None:
    with pytest.raises(ValidationError):
        PhotoAugmentConfig(gaussian_blur_p=p)


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_photoaugmentconfig_noise_probability_out_of_range_raises(p: float) -> None:
    with pytest.raises(ValidationError):
        PhotoAugmentConfig(gaussian_noise_p=p)
