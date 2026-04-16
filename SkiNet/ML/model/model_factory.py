from __future__ import annotations

from SkiNet.ML.configs.model_configs.unet2d_config import UNet2DModelConfig
from SkiNet.ML.model.architecture.unet2d import UNet2D
from SkiNet.ML.configs.experiment_config import ExperimentConfig


def create_model(main_config: ExperimentConfig) -> UNet2D:
    """
    Build a model instance from the experiment configuration.

    :param main_config: ExperimentConfig containing the model configuration to build from.
    :return: An instance of the model specified in the experiment configuration.
    """
    model_cfg = main_config.modelconfig

    if isinstance(model_cfg, UNet2DModelConfig):
        return UNet2D(
            in_channels=model_cfg.in_channels,
            out_channels_layer1=model_cfg.out_channels_layer1,
            kernel=model_cfg.kernel,
            stride=model_cfg.stride,
            dilation=model_cfg.dilation,
            number_of_layers=model_cfg.number_of_layers,
            num_output_classes=model_cfg.num_output_classes,
            model_name=model_cfg.model_name,
            validate_forward=model_cfg.validate_forward,
        )

    raise ValueError(
        f"Unsupported model configuration type: {type(model_cfg).__name__}"
    )
