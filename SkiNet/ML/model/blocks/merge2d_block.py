from typing import Callable, cast

from torch import Tensor, nn

from SkiNet.ML.model.blocks.conv2d_layer import Conv2dLayer
from SkiNet.ML.utils.sampling.base_sampling import EncoderParams2D


class Merge2DBlock(nn.Module):
    """
    Merge2DBlock merges the output of a decoder with a skip connection.

    :param in_channels: Number of input channels to the decoder block
    :param out_channels: Number of output channels from the merge block, should match skip connection channels
    :param kernel: Kernel size of the convolution operation (required for Conv2dLayer)
    :param apply_batchnorm: If True, applies batch normalization after adding convolved inputs and in the last Conv2dLayer.
        When enabled, the `apply_bias` parameter is ignored and set to False.
        Default is True
        Note, the Conv2dLayers applied to the inputs do not use batch normalization.
        Instead, the batch normalization is explicitly applied after adding the convolved inputs.
    :param apply_bias: If True, applies a learnable bias to the convolution output in the last Conv2dLayer.
        Default is False
    :param activation: Non-linear activation function applied after batch normalization in the merge block and in the last Conv2dLayer.
        Note, the Conv2dLayers applied to the inputs do not use activation functions.
        Instead, the activation function is explicitly applied after adding the convolved inputs and batch normalization.
    """

    def __init__(self,
                 layer_number: int,
                 in_channels: int,
                 out_channels: int,
                 conv_params: EncoderParams2D,
                 activation: Callable[[], nn.Module] = nn.ReLU):
        super().__init__()
        self.layer_number = layer_number

        self.merging_layer = True
        """Denotes if the layer merges the output of a decoder with a skip connection. Required for the forward method of the UNet."""

        self.conv1 = Conv2dLayer(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel=conv_params.kernel,
                                 stride=(1, 1),
                                 dilation=conv_params.dilation,
                                 padding=conv_params.padding,
                                 apply_bias=True,
                                 apply_batchnorm=False,
                                 activation=None)

        self.conv2 = Conv2dLayer(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel=conv_params.kernel,
                                 stride=(1, 1),
                                 dilation=conv_params.dilation,
                                 padding=conv_params.padding,
                                 apply_bias=True,
                                 apply_batchnorm=False,
                                 activation=None)

        self.batchnorm2d = nn.BatchNorm2d(out_channels)

        self.conv3 = Conv2dLayer(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel=conv_params.kernel,
                                 stride=(1, 1),
                                 dilation=conv_params.dilation,
                                 padding=conv_params.padding,
                                 apply_bias=False,
                                 apply_batchnorm=True,
                                 activation=activation)

        self.activation = activation()

    def forward(self, x: Tensor, skip_connection_map: Tensor) -> Tensor:
        x = self.conv1(x) + self.conv2(skip_connection_map)
        x = self.batchnorm2d(x)
        x = self.activation(x)
        return cast(Tensor, self.conv3(x) + x)
