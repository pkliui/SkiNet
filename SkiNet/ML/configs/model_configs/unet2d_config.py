from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from SkiNet.ML.configs.model_configs.base_model_config import BaseModelConfig
from SkiNet.ML.utils.sampling.encoder_sampling import validate_conv_inputs
from SkiNet.ML.utils.typing_utils import IntOrTuple2d


class UNet2DModelConfig(BaseModelConfig):
    """
    Architecture config for UNet2D.

    :param in_channels: Number of input channels.
    :param out_channels_layer1: Number of output channels in the 1st layer of the encoder.
    :param number_of_layers: Number of layers in the encoder path. Default is 5.
        The count starts from layer 1, which is the shallowest layer.
        The number of decoder layers is number_of_layers - 1 and there is one more additional last convolutional layer.
    :param num_output_classes: Number of output classes for segmentation. Default is 1.
    :param kernel: Kernel size of the convolution operation. Default is 3.
    :param stride: Stride of the convolution operation. If not 1, it acts as a downsampling factor in encoder layers and as
        an upsampling factor in decoder layers. Default is 2.
    :param dilation: Dilation factor of the convolution operation. Default is 1.
    :param encoder_residual_mode: Residual mode used in encoder blocks. Default is "he2".
    :param merge_residual_mode: Residual mode used in merge blocks. Default is "he2".
    :param model_name: Name of the model.
    :param validate_forward: If True, perform structural validation checks (skip keys/count) during the forward pass. Default is True.
    :param debug_forward: If True, log warnings for near-zero skip connections.
        Runs tensor reductions on GPU every step — keep False in production. Default is False.
    """

    kind: Literal["unet2d"] = "unet2d"

    # required architecture params
    in_channels: int = Field(default=3, le=3, ge=3)
    out_channels_layer1: int = Field(default=16, ge=1)
    number_of_layers: int = Field(default=5, ge=2)
    num_output_classes: int = Field(default=1, ge=1)

    # conv hyperparams
    kernel: IntOrTuple2d = 3
    stride: IntOrTuple2d = 2
    dilation: IntOrTuple2d = 1

    # residual modes
    encoder_residual_mode: Literal["classical", "local_refinement", "he2", "se"] = "he2"
    merge_residual_mode: Literal["classical", "local_refinement", "he1", "he2", "attention_gate"] = "he2"
    se_reduction: int = Field(default=16, ge=1)

    # runtime / debugging
    model_name: str = "UNet2D"
    validate_forward: bool = True
    debug_forward: bool = False

    @model_validator(mode="after")
    def _validate_model_inputs(self) -> "UNet2DModelConfig":
        """
        Validate the inputs of the UNet2D model.
        """
        validate_conv_inputs(kernel=self.kernel, dilation=self.dilation, stride=self.stride)
        return self

    @property
    def number_of_downsampling_layers(self) -> int:
        """
        The number of downsampling layers in the encoder path as per model design in UNet2D._build_encoders
        """
        return self.number_of_layers - 1

    @property
    def required_input_multiple(self) -> IntOrTuple2d:
        """
        Required input height/width must be divisible by the cumulative
        downsampling factor of the encoder. For a model with stride=2
        downsampling applied `n_downsampling_layers` times, this is
        `2 ** n_downsampling_layers`.
        """
        n = self.number_of_downsampling_layers
        s = self.stride

        if isinstance(s, tuple):
            stride_h, stride_w = s
            return (stride_h ** n, stride_w ** n)

        # mypy: s is int here
        if isinstance(s, int):
            return s ** n  # type: ignore[no-any-return]
        # mypy
        raise TypeError(f"stride must be int or tuple[int, int], got {type(s).__name__}")
