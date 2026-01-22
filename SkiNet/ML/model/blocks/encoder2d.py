from typing import Callable, Optional, cast

import torch.nn as nn
from torch import Tensor

from SkiNet.ML.model.blocks.conv2d_layer import Conv2dLayer
from SkiNet.ML.utils.sampling.base_sampling import EncoderParams2D


class Encoder2D(nn.Module):
    """
    Encoder2D is an encoder block composed of a downsampling and a shape-preserving convolutional layers

    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param conv_params: An `EncoderParams2D` instance with validated kernel, stride, dilation, and computed padding.
    :param apply_bias: If True, adds a learnable bias to the output. Note that this parameter is ignored and set to False in Conv2dLayer
        when `apply_batchnorm=True` because batch normalization includes its own learnable parameters.
    :param apply_batchnorm: If True, applies batch normalization after convolution.
    :param activation: Non-linear activation function applied after batch normalization. If None, no activation is applied.
    :param use_residual: If True, enables a skip connection that adds the first convolution's output to the final output of the block.
    :param layer_number: The layer number within the encoder block. The upper-most layer is 1.

    :return: Output tensor after applying convolution, optional batch normalization, and optional activation.
    """

    def __init__(self,
                 layer_number: int,
                 in_channels: int,
                 out_channels: int,
                 conv_params: EncoderParams2D,
                 apply_bias: bool,
                 apply_batchnorm: bool,
                 activation: Callable[[], nn.Module],
                 use_residual: bool):
        super().__init__()

        self.use_residual = use_residual
        self.layer_number = layer_number
        self.merging_layer = False
        """Denotes if the layer merges the output of a decoder with a skip connection. Required for the forward method of the UNet."""

        # downsampling convolutional layer
        self.conv2d_layer1 = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=conv_params.kernel,
            stride=conv_params.stride,
            dilation=conv_params.dilation,
            padding=conv_params.padding,
            apply_bias=apply_bias,
            apply_batchnorm=apply_batchnorm,
            activation=activation)

        # use stride = 1 in the second layer to keep the size of the downsampled output
        self.conv2d_layer2 = Conv2dLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel=conv_params.kernel,
            stride=(1, 1),
            dilation=conv_params.dilation,
            padding=conv_params.padding,
            apply_bias=apply_bias,
            apply_batchnorm=apply_batchnorm,
            activation=activation)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv2d_layer1(x)
        conv2 = self.conv2d_layer2(x)
        # cast for mypy
        return cast(Tensor, conv2 + x if self.use_residual else conv2)
