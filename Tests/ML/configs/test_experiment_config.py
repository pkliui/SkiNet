from pathlib import Path

import pytest
from pydantic import ValidationError

from SkiNet.ML.configs.data_configs.ph2dataset_config.ph2dataset_config import PH2DatasetConfig
from SkiNet.ML.configs.experiment_config import ExperimentConfig
from SkiNet.ML.configs.model_configs.unet2d_config import UNet2DModelConfig
from SkiNet.ML.configs.train_configs.base_train_config import BaseTrainConfig
from SkiNet.ML.configs.transform_configs.crop_config import CropConfig
from SkiNet.ML.configs.transform_configs.transform_config import TransformConfig


def make_valid_experiment_config_kwargs() -> dict:
    return {"experiment_type": "segmentation",
            "experiment_name": "unet2d_ph2_experiment",
            "description": "UNet2D on PH2 dataset",
            # provide minimal placeholder values for required PH2DatasetConfig args
            "dataconfig": PH2DatasetConfig(
                azure_data=False,
                azure_blob_mount_point=None,
                local_data_root=str(Path("/tmp")),
                crop_size=(256, 256),
                kind="ph2"),
            "transformconfig": TransformConfig(),
            "modelconfig": UNet2DModelConfig(),
            "trainconfig": BaseTrainConfig()}


def test_experiment_config_valid() -> None:
    """
    ExperimentConfig should be created successfully with valid nested configs.
    """
    config = ExperimentConfig(**make_valid_experiment_config_kwargs())

    assert config.experiment_type == "segmentation"
    assert config.experiment_name == "unet2d_ph2_experiment"
    assert config.description == "UNet2D on PH2 dataset"
    assert isinstance(config.dataconfig, PH2DatasetConfig)
    assert isinstance(config.transformconfig, TransformConfig)
    assert isinstance(config.modelconfig, UNet2DModelConfig)
    assert isinstance(config.trainconfig, BaseTrainConfig)


def test_experiment_config_forbids_extra_top_level_fields() -> None:
    """
    ExperimentConfig should reject unknown top-level fields.
    """
    kwargs = make_valid_experiment_config_kwargs()
    kwargs["unexpected_field"] = "not allowed"

    with pytest.raises(ValidationError, match="unexpected_field"):
        ExperimentConfig(**kwargs)


@pytest.mark.parametrize(
    "missing_field",
    [
        "experiment_type",
        "experiment_name",
        "description",
        "dataconfig",
        "transformconfig",
        "modelconfig",
        "trainconfig",
    ],
)
def test_experiment_config_missing_required_fields(missing_field: str) -> None:
    """
    ExperimentConfig should raise ValidationError when required fields are missing.
    """
    kwargs = make_valid_experiment_config_kwargs()
    del kwargs[missing_field]

    with pytest.raises(ValidationError, match=missing_field):
        ExperimentConfig(**kwargs)


def test_experiment_config_accepts_valid_nested_transform_override() -> None:
    """
    ExperimentConfig should keep a provided TransformConfig instance and expose its nested override values.
    """
    kwargs = make_valid_experiment_config_kwargs()
    kwargs["transformconfig"] = TransformConfig(
        augmentation_required=False,
        seed_value=13,
        crop=CropConfig(crop_type="center_crop", size=(64, 48)),
    )

    config = ExperimentConfig(**kwargs)

    assert isinstance(config.transformconfig, TransformConfig)
    assert config.transformconfig.augmentation_required is False
    assert config.transformconfig.seed_value == 13
    assert config.transformconfig.crop.crop_type == "center_crop"
    assert config.transformconfig.crop.size == (64, 48)


def test_experiment_config_validates_crop_multiple_against_model_ok() -> None:
    kwargs = make_valid_experiment_config_kwargs()
    kwargs["modelconfig"] = UNet2DModelConfig(number_of_layers=5, stride=2)
    kwargs["transformconfig"] = TransformConfig(
        crop=CropConfig(crop_apply=True, crop_type="center_crop", size=(64, 48))
    )
    cfg = ExperimentConfig(**kwargs)
    assert cfg.transformconfig.crop.size == (64, 48)


def test_experiment_config_validates_crop_multiple_against_model_raises() -> None:
    kwargs = make_valid_experiment_config_kwargs()
    kwargs["modelconfig"] = UNet2DModelConfig(number_of_layers=5, stride=2)  # multiple=16
    kwargs["transformconfig"] = TransformConfig(
        crop=CropConfig(crop_apply=True, crop_type="center_crop", size=(62, 48))
    )
    with pytest.raises(ValidationError):
        ExperimentConfig(**kwargs)
