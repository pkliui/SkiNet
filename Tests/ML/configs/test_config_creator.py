import pytest
from pydantic import ValidationError

from SkiNet.ML.configs.config_creator import ISIC2017_UNet_ConfigCreator, PH2_UNet_ConfigCreator
from SkiNet.ML.configs.data_configs.isic2017dataset_config.isic2017dataset_config import ISIC2017DatasetConfig
from SkiNet.ML.configs.data_configs.ph2dataset_config.ph2dataset_config import PH2DatasetConfig
from SkiNet.ML.configs.experiment_config import ExperimentConfig
from SkiNet.ML.configs.model_configs.unet2d_config import UNet2DModelConfig
from SkiNet.ML.configs.train_configs.train_config import TrainConfig
from SkiNet.ML.configs.transform_configs.transform_config import TransformConfig
from SkiNet.Utils.experiment_keys import ExperimentType
from SkiNet.ML.configs.train_configs.sweep_config import SweepConfig


def test_ph2_unet_config_creator_returns_experiment_config_default() -> None:
    """
    create_config() with no kwargs returns a fully formed ExperimentConfig
    with expected metadata and config object types.
    """
    creator = PH2_UNet_ConfigCreator()
    config = creator.create_config()

    assert isinstance(config, ExperimentConfig)
    assert config.experiment_name == "unet2d_ph2_experiment"
    assert config.experiment_type == ExperimentType.SEGMENTATION
    assert config.description == "UNet2D on PH2 dataset"

    assert isinstance(config.dataconfig, PH2DatasetConfig)
    assert isinstance(config.modelconfig, UNet2DModelConfig)
    assert isinstance(config.trainconfig, TrainConfig)
    assert isinstance(config.transformconfig, TransformConfig)
    assert isinstance(config.sweepconfig, SweepConfig)


@pytest.mark.parametrize(
    "dataconfig_kwargs,transformconfig_kwargs,modelconfig_kwargs,trainconfig_kwargs,sweepconfig_kwargs",
    [
        ({}, {}, {}, {}, {}),
        (None, None, None, None, None),
    ],
)
def test_ph2_unet_config_creator_accepts_empty_or_none_kwargs(
    dataconfig_kwargs: dict | None,
    transformconfig_kwargs: dict | None,
    modelconfig_kwargs: dict | None,
    trainconfig_kwargs: dict | None,
    sweepconfig_kwargs: dict | None,
) -> None:
    """
    None kwargs should be normalized to empty dicts and still produce valid config objects.
    """
    creator = PH2_UNet_ConfigCreator()
    config = creator.create_config(
        dataconfig_kwargs=dataconfig_kwargs,
        transformconfig_kwargs=transformconfig_kwargs,
        modelconfig_kwargs=modelconfig_kwargs,
        trainconfig_kwargs=trainconfig_kwargs,
        sweepconfig_kwargs=sweepconfig_kwargs,
    )

    assert isinstance(config, ExperimentConfig)
    assert isinstance(config.dataconfig, PH2DatasetConfig)
    assert isinstance(config.transformconfig, TransformConfig)
    assert isinstance(config.modelconfig, UNet2DModelConfig)
    assert isinstance(config.trainconfig, TrainConfig)
    assert isinstance(config.sweepconfig, SweepConfig)


@pytest.mark.parametrize(
    "kwargs_type,extra_kwargs,should_raise",
    [
        ("dataconfig", {"__invalid_arg__": 1}, False),   # PH2DatasetConfig ignores extra fields
        ("transformconfig", {"__invalid_arg__": 1}, False),  # TransformConfig ignores extra fields
        ("modelconfig", {"__invalid_arg__": 1}, True),   # UNet2DModelConfig forbids extra fields
        ("trainconfig", {"__invalid_arg__": 1}, True),   # TrainConfig forbids extra fields
        ("sweepconfig", {"__invalid_arg__": 1}, True),   # SweepConfig forbids extra fields
    ],
)
def test_ph2_unet_config_creator_unknown_kwargs_behavior(
    kwargs_type: str,
    extra_kwargs: dict,
    should_raise: bool,
) -> None:
    """
    Unknown kwargs should raise ValidationError for configs with extra='forbid'
    (UNet2DModelConfig, TrainConfig) and be silently ignored for configs with extra='ignore'.
    """
    creator = PH2_UNet_ConfigCreator()
    kwargs = {f"{kwargs_type}_kwargs": extra_kwargs}

    if should_raise:
        with pytest.raises(ValidationError):
            creator.create_config(**kwargs)  # type: ignore[arg-type]
    else:
        config = creator.create_config(**kwargs)  # type: ignore[arg-type]
        assert isinstance(config, ExperimentConfig)
        sub_config = getattr(config, kwargs_type)
        for key in extra_kwargs:
            assert not hasattr(sub_config, key)


def test_ph2_unet_config_creator_applies_transformconfig_overrides() -> None:
    """
    transformconfig_kwargs should be forwarded into the nested TransformConfig model.
    """
    creator = PH2_UNet_ConfigCreator()

    config = creator.create_config(
        transformconfig_kwargs={
            "augmentation_required": False,
            "seed_value": 7,
            "crop": {"crop_type": "center_crop", "size": (64, 48)},
            "photometric_augmentation": {"color_jitter_apply": False},
        }
    )

    assert config.transformconfig.augmentation_required is False
    assert config.transformconfig.seed_value == 7
    assert config.transformconfig.crop.crop_type == "center_crop"
    assert config.transformconfig.crop.size == (64, 48)
    assert config.transformconfig.photometric_augmentation.color_jitter_apply is False


def test_ph2_unet_config_creator_custom_experiment_name() -> None:
    """
    experiment_name kwarg should override the hardcoded default.
    """
    creator = PH2_UNet_ConfigCreator()
    config = creator.create_config(experiment_name="my_custom_run")

    assert config.experiment_name == "my_custom_run"


def test_ph2_unet_config_creator_default_experiment_name_unchanged() -> None:
    """
    Omitting experiment_name should still produce the original default value.
    """
    creator = PH2_UNet_ConfigCreator()
    config = creator.create_config()

    assert config.experiment_name == "unet2d_ph2_experiment"


def test_ph2_unet_config_creator_uses_default_transformconfig_when_not_overridden() -> None:
    """
    create_config() should use TransformConfig model defaults when no external transformconfig kwargs are provided.
    """
    creator = PH2_UNet_ConfigCreator()

    config = creator.create_config()

    assert isinstance(config.transformconfig, TransformConfig)
    assert config.transformconfig.augmentation_required is True
    assert config.transformconfig.seed_value is None
    assert config.transformconfig.crop.crop_apply is True
    assert config.transformconfig.crop.crop_type == "random_resized_crop"
    assert config.transformconfig.crop.size == (512, 512)
    assert config.transformconfig.crop.scale == (0.8, 1.0)
    assert config.transformconfig.photometric_augmentation.color_jitter_apply is False
    assert config.transformconfig.photometric_augmentation.color_jitter_p == 0.5


# ======================================================================
# ISIC2017_UNet_ConfigCreator
# ======================================================================

def test_isic2017_unet_config_creator_returns_experiment_config_default() -> None:
    creator = ISIC2017_UNet_ConfigCreator()
    config = creator.create_config()

    assert isinstance(config, ExperimentConfig)
    assert config.experiment_name == "unet2d_isic2017_experiment"
    assert config.experiment_type == ExperimentType.SEGMENTATION
    assert config.description == "UNet2D on ISIC2017 dataset"

    assert isinstance(config.dataconfig, ISIC2017DatasetConfig)
    assert isinstance(config.modelconfig, UNet2DModelConfig)
    assert isinstance(config.trainconfig, TrainConfig)
    assert isinstance(config.transformconfig, TransformConfig)
    assert isinstance(config.sweepconfig, SweepConfig)


@pytest.mark.parametrize(
    "dataconfig_kwargs,transformconfig_kwargs,modelconfig_kwargs,trainconfig_kwargs,sweepconfig_kwargs",
    [
        ({}, {}, {}, {}, {}),
        (None, None, None, None, None),
    ],
)
def test_isic2017_unet_config_creator_accepts_empty_or_none_kwargs(
    dataconfig_kwargs: dict | None,
    transformconfig_kwargs: dict | None,
    modelconfig_kwargs: dict | None,
    trainconfig_kwargs: dict | None,
    sweepconfig_kwargs: dict | None,
) -> None:
    creator = ISIC2017_UNet_ConfigCreator()
    config = creator.create_config(
        dataconfig_kwargs=dataconfig_kwargs,
        transformconfig_kwargs=transformconfig_kwargs,
        modelconfig_kwargs=modelconfig_kwargs,
        trainconfig_kwargs=trainconfig_kwargs,
        sweepconfig_kwargs=sweepconfig_kwargs,
    )

    assert isinstance(config, ExperimentConfig)
    assert isinstance(config.dataconfig, ISIC2017DatasetConfig)
    assert isinstance(config.transformconfig, TransformConfig)
    assert isinstance(config.modelconfig, UNet2DModelConfig)
    assert isinstance(config.trainconfig, TrainConfig)
    assert isinstance(config.sweepconfig, SweepConfig)


def test_isic2017_unet_config_creator_custom_experiment_name() -> None:
    creator = ISIC2017_UNet_ConfigCreator()
    config = creator.create_config(experiment_name="isic2017_custom_run")
    assert config.experiment_name == "isic2017_custom_run"


def test_isic2017_unet_config_creator_default_experiment_name_unchanged() -> None:
    creator = ISIC2017_UNet_ConfigCreator()
    config = creator.create_config()
    assert config.experiment_name == "unet2d_isic2017_experiment"
