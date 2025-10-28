import torch
import torch.nn as nn
from typing import Optional, Callable

from SkiNet.ML.model.tr_conv2d_layer import TrConv2dLayer
from SkiNet.ML.utils.sampling.decoder_sampling import EncoderSpec


class Decoder2D(nn.Module):
    """
    Decoder2D is a decoder block that consists of a TrConv2dLayer.

    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param encoder_spec: EncoderSpec describing the corresponding encoder convolution to be inverted.
    :param activation: Non-linear activation function applied after batch normalization. If None, no activation is applied.
        Default is torch.nn.ReLU
    """
    def __init__(self,
                 in_channels: int,
                 out_channels:  int,
                 encoder_spec: EncoderSpec,
                 activation: Optional[Callable] = torch.nn.ReLU):
        super().__init__()

        self.decoder_layer = TrConv2dLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    encoder_spec=encoder_spec,
                    activation=activation)

    def forward(self, x):
        x = self.decoder_layer(x)
        return x