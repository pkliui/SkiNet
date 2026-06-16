import logging

import pytest

from SkiNet.ML.configs.config_creator import ISIC2017_UNet_ConfigCreator, PH2_UNet_ConfigCreator
from SkiNet.ML.configs.config_factory import (
    ConfigFactory,
    ISI2017_UNet_ConfigFactory,
    PH2_UNet_ConfigFactory,
    get_config_factory,
)
from SkiNet.Utils.experiment_keys import DatasetKey, ModelKey


def test_config_factory_is_abstract() -> None:
    """
    ConfigFactory is abstract and cannot be instantiated directly.
    """
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        ConfigFactory()  # type: ignore[abstract]


@pytest.mark.parametrize(
    "model_key,dataset_key,expected_factory_type",
    [
        (ModelKey.UNET2D, DatasetKey.PH2, PH2_UNet_ConfigFactory),
        (ModelKey.UNET2D, DatasetKey.ISIC2017, ISI2017_UNet_ConfigFactory),
    ],
)
def testget_config_factory_valid(model_key: ModelKey, dataset_key: DatasetKey, expected_factory_type: type) -> None:
    """
    Test that the correct config factory is returned for valid model and dataset keys.
    """
    factory = get_config_factory(model_key, dataset_key)
    assert isinstance(factory, expected_factory_type)


@pytest.mark.parametrize(
    "model_key,dataset_key",
    [
        ("unknown", DatasetKey.PH2),
        (ModelKey.UNET2D, "unknown"),
        ("unknown", "unknown"),
    ],
)
def testget_config_factory_invalid_raises_and_logs(model_key: ModelKey, dataset_key: DatasetKey, caplog: pytest.LogCaptureFixture) -> None:
    """
    Test that the correct error is raised and logged for invalid model and dataset keys.
    """
    logger_name = "SkiNet.ML.configs.config_factory"
    with caplog.at_level(logging.ERROR, logger=logger_name):
        with pytest.raises(ValueError) as exc_info:
            get_config_factory(model_key, dataset_key)

    msg = str(exc_info.value)
    assert "No factory found for model key:" in msg
    assert "dataset key:" in msg
    assert str(model_key) in msg
    assert str(dataset_key) in msg
    assert any(record.name == logger_name
               and "No factory found for model key:" in record.message
               and "dataset key:" in record.message
               for record in caplog.records)


def testget_config_factory_returns_fresh_instances() -> None:
    """
    Ensure we instantiate a fresh factory per call (avoid shared mutable state).
    """
    f1 = get_config_factory(ModelKey.UNET2D, DatasetKey.PH2)
    f2 = get_config_factory(ModelKey.UNET2D, DatasetKey.PH2)
    assert f1 is not f2


def test_factory_get_config_creator_returns_expected_creator() -> None:
    """
    Ensure the returned factory creates the expected config creator.
    """
    factory = get_config_factory(ModelKey.UNET2D, DatasetKey.PH2)
    creator = factory.get_config_creator()
    assert isinstance(creator, PH2_UNet_ConfigCreator)


def test_isic2017_factory_get_config_creator_returns_expected_creator() -> None:
    factory = get_config_factory(ModelKey.UNET2D, DatasetKey.ISIC2017)
    creator = factory.get_config_creator()
    assert isinstance(creator, ISIC2017_UNet_ConfigCreator)
