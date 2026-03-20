from typing import Any
from unittest.mock import MagicMock

import pytest

from SkiNet.ML.configs.data_configs.ph2dataset_config.ph2dataset_config import PH2DatasetConfig
from SkiNet.ML.configs.experiment_config import ExperimentConfig
from SkiNet.ML.configs.model_configs.unet2d_config import UNet2DModelConfig
from SkiNet.ML.configs.train_configs.base_train_config import BaseTrainConfig
from SkiNet.ML.model.architecture.unet2d import UNet2D
from SkiNet.ML.utils.model_factory import create_model


@pytest.fixture
def mock_dataconfig() -> PH2DatasetConfig:
    cfg = MagicMock(spec=PH2DatasetConfig)
    cfg.kind = "ph2"
    return cfg

@pytest.fixture
def mock_trainconfig() -> BaseTrainConfig:
    return MagicMock(spec=BaseTrainConfig)

@pytest.fixture
def mock_modelconfig() -> UNet2DModelConfig:
    cfg = MagicMock(spec=UNet2DModelConfig)
    cfg.kind = "unet2d"
    return cfg

@pytest.mark.parametrize(
    "model_cfg, expected_type, expected_attrs",
    [
        (
            UNet2DModelConfig(in_channels=1,
                              out_channels_layer1=4,
                              kernel=3,
                              stride=2,
                              dilation=1,
                              number_of_layers=5,
                              num_output_classes=1,
                              model_name="UNet2D",
                              validate_forward=False),
            UNet2D,
            {
                "in_channels": 1,
                "out_channels_layer1": 4,
                "kernel": 3,
                "stride": 2,
                "dilation": 1,
                "number_of_layers": 5,
                "num_output_classes": 1,
                "model_name": "UNet2D",
                "validate_forward": False
            }
        ),
        (
            UNet2DModelConfig(in_channels=3,
                              out_channels_layer1=8,
                              kernel=(3, 3),
                              stride=(2, 2),
                              dilation=(1, 1),
                              number_of_layers=4,
                              num_output_classes=2,
                              model_name="UNet2D",
                              validate_forward=True),
            UNet2D,
            {
                "in_channels": 3,
                "out_channels_layer1": 8,
                "kernel": (3, 3),
                "stride": (2, 2),
                "dilation": (1, 1),
                "number_of_layers": 4,
                "num_output_classes": 2,
                "model_name": "UNet2D",
                "validate_forward": True,
            },
        ),
    ],
)
def test_create_model(model_cfg: UNet2DModelConfig,
                      expected_type: type,
                      expected_attrs: dict,
                      mock_dataconfig: PH2DatasetConfig,
                      mock_trainconfig: BaseTrainConfig) -> None:
    """
    Test the model creation factory function with various UNet2D model configurations.
    """
    # Bypass ExperimentConfig validation: we only want to test the factory logic.
    cfg = ExperimentConfig(
        experiment_type="classification",
        experiment_name="test_experiment",
        description="Test experiment",
        dataconfig=mock_dataconfig,
        trainconfig=mock_trainconfig,
        modelconfig=model_cfg
    )
    model = create_model(cfg)

    assert isinstance(model, expected_type)
    for attr, expected in expected_attrs.items():
        assert getattr(model, attr) == expected, f"Mismatch for attribute '{attr}'"


@pytest.mark.parametrize("bad_model_cfg", [
    object(), "not a model config", 123, {"kind": "unknown"}
])
def test_create_model_raises_for_unsupported_model_config_type(bad_model_cfg: Any,
                                                               mock_trainconfig: BaseTrainConfig,
                                                               mock_dataconfig: PH2DatasetConfig) -> None:
    """
    Test the model creation factory function with various invalid model configurations.
    """
    with pytest.raises(ValueError, match="validation error"):

        cfg = ExperimentConfig(
            experiment_type="classification",
            experiment_name="test_experiment",
            description="Test experiment",
            dataconfig=mock_dataconfig,
            trainconfig=mock_trainconfig,
            modelconfig=bad_model_cfg
        )
        create_model(cfg)
