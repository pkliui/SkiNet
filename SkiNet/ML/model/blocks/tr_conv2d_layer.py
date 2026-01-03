from typing import Callable

import torch.nn as nn
from torch import Tensor

from SkiNet.ML.utils.sampling.decoder_sampling import get_output_padding, get_padding_for_transpose_conv
from SkiNet.ML.utils.typing_utils import IntOrTuple2d


class TrConv2dLayer(nn.Module):
    """
    TrConv2dLayer is a 2D transposed convolution followed by batch normalization

    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param decoding_kernel_size: Kernel size for transposed convolution
    :param decoding_stride: Stride for transposed convolution
    :param activation: Non-linear activation function applied after batch normalization.
        Default is torch.nn.ReLU.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 decoding_kernel_size: IntOrTuple2d,
                 decoding_stride: IntOrTuple2d,
                 activation: Callable = nn.ReLU):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.deconv2d = nn.ConvTranspose2d(in_channels=self.in_channels,
                                           out_channels=self.out_channels,
                                           kernel_size=decoding_kernel_size,
                                           stride=decoding_stride,
                                           padding=get_padding_for_transpose_conv(decoding_kernel_size),
                                           output_padding=get_output_padding(decoding_kernel_size, decoding_stride))

        self.batchnorm2d = nn.BatchNorm2d(num_features=self.out_channels)
        self.activation = activation(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.deconv2d(x)
        x = self.batchnorm2d(x)
        x = self.activation(x)
        return x
