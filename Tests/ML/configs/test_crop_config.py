import pytest
from pydantic import ValidationError
from typing import Literal
from SkiNet.ML.configs.transform_configs.crop_config import CropConfig


def test_cropconfig_defaults() -> None:
    cfg = CropConfig()
    assert cfg.crop_apply is True
    assert cfg.crop_type in {"center_crop", "random_crop", "random_resized_crop"}
    assert isinstance(cfg.size, tuple)
    assert len(cfg.size) == 2
    assert cfg.size[0] > 0 and cfg.size[1] > 0
    assert isinstance(cfg.scale, tuple)
    assert len(cfg.scale) == 2


ValidCropType = Literal["center_crop", "random_crop", "random_resized_crop"]


@pytest.mark.parametrize("crop_type", ["center_crop", "random_crop", "random_resized_crop"])
def test_cropconfig_accepts_valid_crop_types(crop_type: ValidCropType) -> None:
    cfg = CropConfig(crop_apply=True, crop_type=crop_type, size=(64, 32))
    assert cfg.crop_apply is True
    assert cfg.crop_type == crop_type
    assert cfg.size == (64, 32)


def test_cropconfig_allows_overrides() -> None:
    cfg = CropConfig(
        crop_apply=True,
        crop_type="random_resized_crop",
        size=(512, 512),
        scale=(0.8, 1.0),
    )
    assert cfg.crop_apply is True
    assert cfg.crop_type == "random_resized_crop"
    assert cfg.size == (512, 512)
    assert cfg.scale == (0.8, 1.0)


@pytest.mark.parametrize("size", [(0, 32), (32, 0), (-1, 32), (32, -1)])
def test_cropconfig_invalid_size_raises(size: tuple[int, int]) -> None:
    # Requires CropConfig to enforce positive sizes (common via Field(gt=0) or validator).
    with pytest.raises(ValidationError):
        CropConfig(crop_apply=True, crop_type="center_crop", size=size)


@pytest.mark.parametrize("scale", [(-0.1, 1.0), (0.0, 1.0), (1.1, 1.2)])
def test_cropconfig_scale_out_of_range_raises(scale: tuple[float, float]) -> None:
    # Requires CropConfig.scale to be constrained to 0<scale<=1 (or whatever your model enforces).
    with pytest.raises(ValidationError):
        CropConfig(crop_apply=True, crop_type="random_resized_crop", size=(64, 64), scale=scale)


def test_cropconfig_scale_min_greater_than_max_raises() -> None:
    # Requires a validator enforcing min<=max for scale.
    with pytest.raises(ValidationError):
        CropConfig(crop_apply=True, crop_type="random_resized_crop", size=(64, 64), scale=(0.9, 0.2))
