import pytest
from pydantic import ValidationError

from SkiNet.ML.configs.config_creator import PH2_UNet_ConfigCreator
from SkiNet.ML.configs.data_configs.ph2dataset_config.ph2dataset_config import PH2DatasetConfig
from SkiNet.ML.configs.experiment_config import ExperimentConfig
from SkiNet.ML.configs.model_configs.unet2d_config import UNet2DModelConfig
from SkiNet.ML.configs.train_configs.base_train_config import BaseTrainConfig


def test_ph2_unet_config_creator_returns_experiment_config_default() -> None:
    """
    create_config() with no kwargs returns a fully formed ExperimentConfig
    with expected metadata and config object types.
    """
    creator = PH2_UNet_ConfigCreator()
    config = creator.create_config()

    assert isinstance(config, ExperimentConfig)
    assert config.experiment_name == "unet2d_ph2_experiment"
    assert config.experiment_type == "segmentation"
    assert config.description == "UNet2D on PH2 dataset"

    assert isinstance(config.dataconfig, PH2DatasetConfig)
    assert isinstance(config.modelconfig, UNet2DModelConfig)
    assert isinstance(config.trainconfig, BaseTrainConfig)


@pytest.mark.parametrize(
    "dataconfig_kwargs,modelconfig_kwargs,trainconfig_kwargs",
    [
        ({}, {}, {}),
        (None, None, None),
    ],
)
def test_ph2_unet_config_creator_accepts_empty_or_none_kwargs(
    dataconfig_kwargs: dict | None,
    modelconfig_kwargs: dict | None,
    trainconfig_kwargs: dict | None,
) -> None:
    """
    None kwargs should be normalized to empty dicts and still produce valid config objects.
    """
    creator = PH2_UNet_ConfigCreator()
    config = creator.create_config(
        dataconfig_kwargs=dataconfig_kwargs,
        modelconfig_kwargs=modelconfig_kwargs,
        trainconfig_kwargs=trainconfig_kwargs,
    )

    assert isinstance(config, ExperimentConfig)
    assert isinstance(config.dataconfig, PH2DatasetConfig)
    assert isinstance(config.modelconfig, UNet2DModelConfig)
    assert isinstance(config.trainconfig, BaseTrainConfig)


@pytest.mark.parametrize(
    "kwargs_type,invalid_kwargs,should_raise",
    [
        ("dataconfig", {"__invalid_arg__": 1}, False),  # PH2DatasetConfig allows extra fields
        ("modelconfig", {"__invalid_arg__": 1}, True),   # UNet2DModelConfig forbids extra fields
        ("trainconfig", {"__invalid_arg__": 1}, False),  # BaseTrainConfig allows extra fields
    ],
)
def test_ph2_unet_config_creator_invalid_kwargs(kwargs_type: str, invalid_kwargs: dict, should_raise: bool) -> None:
    """
    Test that passing invalid (extra) kwargs raises ValidationError only where configured to forbid extras.
    This checks all config fields, documenting which ones enforce strict validation.
    """
    creator = PH2_UNet_ConfigCreator()
    kwargs = {f"{kwargs_type}_kwargs": invalid_kwargs}

    if should_raise:
        with pytest.raises(ValidationError):
            creator.create_config(**kwargs)
    else:
        # Should not raise; config should still be created successfully
        config = creator.create_config(**kwargs)
        assert isinstance(config, ExperimentConfig)


# --------------------------------Test valid kwargs-------------------------


@pytest.mark.parametrize(
    "kwargs_type,extra_kwargs,should_raise",
    [
        ("dataconfig", {"__invalid_arg__": 1}, False),
        ("modelconfig", {"__invalid_arg__": 1}, True),
        ("trainconfig", {"__invalid_arg__": 1}, False),
    ],
)
def test_ph2_unet_config_creator_unknown_kwargs_behavior(kwargs_type: str, extra_kwargs: dict, should_raise: bool) -> None:
    """
    Test behavior for unknown kwargs:
    - modelconfig forbids extra fields and should raise ValidationError
    - dataconfig/trainconfig ignore extra fields and should still create a config
    """
    creator = PH2_UNet_ConfigCreator()
    kwargs = {f"{kwargs_type}_kwargs": extra_kwargs}

    if should_raise:
        with pytest.raises(ValidationError):
            creator.create_config(**kwargs)
    else:
        config = creator.create_config(**kwargs)
        assert isinstance(config, ExperimentConfig)

        # Check that the extra kwargs are not present in the created config, confirming they were ignored
        sub_config = getattr(config, kwargs_type)
        for key in extra_kwargs:
            assert not hasattr(sub_config, key)
