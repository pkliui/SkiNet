from typing import Callable, Optional

import torch.nn as nn
from torch import Tensor

from SkiNet.ML.model.blocks.conv2d_layer import Conv2dLayer
from SkiNet.ML.utils.typing_utils import IntOrTuple2d


class Encoder2D(nn.Module):
    """
    Encoder2D is an encoder block

    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param kernel: Kernel size of the convolution operation.
    :param stride: Stride of the convolution operation.
    :param dilation: Dilation factor of the convolution operation.
    :param apply_bias: If True, adds a learnable bias to the output. Note that this parameter is ignored and set to False
        when `apply_batchnorm=True` because batch normalization includes its own learnable parameters.
        Default is False
    :param apply_batchnorm: If True, applies batch normalization after convolution. When enabled, the `bias` parameter
        is ignored and set to False.
    :param activation: Non-linear activation function applied after batch normalization. If None, no activation is applied.
        Default is torch.nn.ReLU
    :param use_residual: If True, adds a residual connection from input to output. Default is True.
    :param layer_number: The layer number within the encoder block. The upper-most layer is 0. Default is 0.

    :return: Output tensor after applying convolution, optional batch normalization, and optional activation.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: IntOrTuple2d = 4,
                 stride: IntOrTuple2d = 1,
                 dilation: IntOrTuple2d = 1,
                 apply_bias: bool = False,
                 apply_batchnorm: bool = True,
                 activation: Optional[Callable] = nn.ReLU,
                 use_residual: bool = True,
                 layer_number: int = 0):
        super().__init__()

        self.use_residual = use_residual
        self.layer_number = layer_number

        # stride as per input value
        self.conv2d_layer1 = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=kernel,
            stride=stride,
            dilation=dilation,
            apply_bias=apply_bias,
            apply_batchnorm=apply_batchnorm,
            activation=activation)

        # use stride = 1 in the second layer to keep dimensions
        self.conv2d_layer2 = Conv2dLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel=kernel,
            stride=1,
            dilation=dilation,
            apply_bias=apply_bias,
            apply_batchnorm=apply_batchnorm,
            activation=activation)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv2d_layer1(x)
        conv2 = self.conv2d_layer2(x)
        return conv2 + x if self.use_residual else conv2
