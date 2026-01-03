from typing import Callable

import torch.nn as nn
from torch import Tensor

from SkiNet.ML.model.blocks.tr_conv2d_layer import TrConv2dLayer
from SkiNet.ML.utils.typing_utils import IntOrTuple2d


class Decoder2D(nn.Module):
    """
    Decoder2D is a decoder block that consists of a TrConv2dLayer.

    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param decoding_kernel_size: Kernel size for transposed convolution
    :param decoding_stride: Stride for transposed convolution
    :param activation: Non-linear activation function applied after batch normalization.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 decoding_kernel_size: IntOrTuple2d,
                 decoding_stride: IntOrTuple2d,
                 activation: Callable = nn.ReLU) -> None:
        super().__init__()

        self.decoder_layer = TrConv2dLayer(in_channels=in_channels,
                                           out_channels=out_channels,
                                           decoding_kernel_size=decoding_kernel_size,
                                           decoding_stride=decoding_stride,
                                           activation=activation)

    def forward(self, x: Tensor) -> Tensor:
        x = self.decoder_layer(x)
        return x
