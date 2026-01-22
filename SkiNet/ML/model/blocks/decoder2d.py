from typing import Callable, Optional, cast

from torch import Tensor, nn

from SkiNet.ML.utils.sampling.base_sampling import DecoderParams2D


class Decoder2D(nn.Module):
    """
    Decoder2D is a transposed convolution layer followed by batch normalization

    :param layer_number: Layer number
    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param decoder_params: Decoder-side transposed convolution parameters
    :param activation: Non-linear activation function applied after batch normalization.
    """

    def __init__(self,
                 layer_number: int,
                 in_channels: int,
                 out_channels: int,
                 decoder_params: DecoderParams2D,
                 activation: Callable[[], nn.Module] = nn.ReLU):
        super().__init__()

        self.layer_number = layer_number
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.decoder_params = decoder_params

        self.merging_layer = False
        """Denotes if the layer merges the output of a decoder with a skip connection. Required for the forward method of the UNet."""

        self.deconv2d = nn.ConvTranspose2d(in_channels=self.in_channels,
                                           out_channels=self.out_channels,
                                           kernel_size=self.decoder_params.kernel,
                                           stride=self.decoder_params.stride,
                                           dilation=self.decoder_params.dilation,
                                           padding=self.decoder_params.padding,
                                           output_padding=self.decoder_params.output_padding)

        self.batchnorm2d = nn.BatchNorm2d(num_features=self.out_channels)
        self.activation = activation()

    def forward(self, x: Tensor) -> Tensor:
        x = self.deconv2d(x)
        x = self.batchnorm2d(x)
        # cast for mypy
        x = cast(Tensor, self.activation(x))
        return x
