from pathlib import Path

import pytest
from pydantic import ValidationError

from SkiNet.ML.configs.data_configs.ph2dataset_config.ph2dataset_config import PH2DatasetConfig
from SkiNet.ML.configs.experiment_config import ExperimentConfig, ExperimentType
from SkiNet.ML.configs.model_configs.unet2d_config import UNet2DModelConfig
from SkiNet.ML.configs.train_configs.train_config import TrainConfig, EarlyStoppingConfig, CheckpointConfig
from SkiNet.ML.configs.train_configs.sweep_config import SweepConfig
from SkiNet.ML.configs.transform_configs.crop_config import CropConfig
from SkiNet.ML.configs.transform_configs.transform_config import TransformConfig
from SkiNet.Utils.experiment_keys import MetricsKey


def make_valid_experiment_config_kwargs() -> dict:
    return {"experiment_type": ExperimentType.SEGMENTATION,
            "experiment_name": "unet2d_ph2_experiment",
            "description": "UNet2D on PH2 dataset",
            # provide minimal placeholder values for required PH2DatasetConfig args
            "dataconfig": PH2DatasetConfig(
                azure_data=False,
                azure_blob_mount_point=None,
                local_data_root=str(Path("/tmp")),
                kind="ph2",
                split_train_size=0.6,
                split_val_size=0.25,
                split_test_size=0.15,
                split_random_seed=42),
            "transformconfig": TransformConfig(),
            "modelconfig": UNet2DModelConfig(),
            "trainconfig": TrainConfig()}


def test_experiment_config_valid() -> None:
    """
    ExperimentConfig should be created successfully with valid nested configs.
    """
    config = ExperimentConfig(**make_valid_experiment_config_kwargs())

    assert config.experiment_type == ExperimentType.SEGMENTATION
    assert config.experiment_name == "unet2d_ph2_experiment"
    assert config.description == "UNet2D on PH2 dataset"
    assert isinstance(config.dataconfig, PH2DatasetConfig)
    assert isinstance(config.transformconfig, TransformConfig)
    assert isinstance(config.modelconfig, UNet2DModelConfig)
    assert isinstance(config.trainconfig, TrainConfig)
    assert isinstance(config.sweepconfig, SweepConfig)


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
    Fields that have a default_factory are not required and are not listed here (e.g. sweepconfig)
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


def test_experiment_config_propagates_sweep_monitor_to_callbacks() -> None:
    """
    sweepconfig.monitor is the single source of truth.  ExperimentConfig must
    copy it into early_stopping_config, checkpoint_config and lr_scheduler_config
    so callers never need to set monitor in three places.
    """
    cfg = ExperimentConfig(**make_valid_experiment_config_kwargs())
    monitor = cfg.sweepconfig.monitor
    assert cfg.trainconfig.early_stopping_config.monitor == monitor
    assert cfg.trainconfig.checkpoint_config.monitor == monitor
    assert cfg.trainconfig.lr_scheduler_config.monitor == monitor


def test_experiment_config_propagates_custom_sweep_monitor() -> None:
    """
    When sweepconfig.monitor is overridden from its default, the propagated
    value in all three callback configs must match the custom value.
    """
    kwargs = make_valid_experiment_config_kwargs()
    kwargs["sweepconfig"] = SweepConfig(monitor=MetricsKey.VAL_BEST_DICE_AT_THRESHOLD)
    cfg = ExperimentConfig(**kwargs)
    assert cfg.trainconfig.early_stopping_config.monitor == MetricsKey.VAL_BEST_DICE_AT_THRESHOLD
    assert cfg.trainconfig.checkpoint_config.monitor == MetricsKey.VAL_BEST_DICE_AT_THRESHOLD
    assert cfg.trainconfig.lr_scheduler_config.monitor == MetricsKey.VAL_BEST_DICE_AT_THRESHOLD


def test_experiment_config_raises_on_monitor_mismatch() -> None:
    """
    If a sub-config has a monitor value that differs from sweepconfig.monitor,
    ExperimentConfig must raise ValueError at construction time rather than
    silently optimising on a different metric than the one used for early stopping.

    In practice this guard only fires when someone sets monitor explicitly in
    both SWEEP_CONFIG and a sub-config in the YAML to different values.
    """
    kwargs = make_valid_experiment_config_kwargs()
    # Artificially create a mismatch: sweepconfig uses the default monitor,
    # but early_stopping_config.monitor is set to a different MetricsKey.
    # Since MetricsKey only has one member today we monkeypatch the validator
    # by providing a sweepconfig whose monitor string differs from the default
    # sub-config monitor via direct TrainConfig construction.
    mismatched_train = TrainConfig(
        early_stopping_config=EarlyStoppingConfig(monitor=MetricsKey.VAL_BEST_DICE_AT_THRESHOLD),
        checkpoint_config=CheckpointConfig(monitor=MetricsKey.VAL_BEST_DICE_AT_THRESHOLD),
    )
    # Patch the sweep monitor to a raw string that won't match MetricsKey members
    # by bypassing SweepConfig validation (use model_construct to skip validators).
    bad_sweep = SweepConfig.model_construct(monitor="some_other_metric", direction="maximize")
    kwargs["trainconfig"] = mismatched_train
    kwargs["sweepconfig"] = bad_sweep
    with pytest.raises((ValidationError, ValueError), match="sweepconfig.monitor"):
        ExperimentConfig(**kwargs)


def test_experiment_config_validates_crop_multiple_against_model_raises() -> None:
    kwargs = make_valid_experiment_config_kwargs()
    kwargs["modelconfig"] = UNet2DModelConfig(number_of_layers=5, stride=2)  # multiple=16
    kwargs["transformconfig"] = TransformConfig(
        crop=CropConfig(crop_apply=True, crop_type="center_crop", size=(62, 48))
    )
    with pytest.raises(ValidationError):
        ExperimentConfig(**kwargs)
