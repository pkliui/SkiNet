from typing import Callable, Iterable, Optional, Union
import torch

from SkiNet.ML.model.conv2d_layer import Conv2dLayer
from SkiNet.ML.utils.sampling.encoder_sampling import PaddingMode

class Merge2DBlock(torch.nn.Module):
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
                    in_channels: int,
                    out_channels:  int,
                    kernel: Union[int, Iterable[int]] = 4,
                    apply_batchnorm: bool = True,
                    apply_bias: bool = False,
                    activation: Optional[Callable] = torch.nn.ReLU):
        super().__init__()

        self.conv1 = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=kernel,
            padding_mode=PaddingMode.SAME,
            apply_batchnorm=False,
            apply_bias=True,
            activation=None
        )
        self.conv2 = Conv2dLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel=kernel,
            padding_mode=PaddingMode.SAME,
            apply_batchnorm=False,
            apply_bias=True,
            activation=None
        )
        self.batchnorm2d = torch.nn.BatchNorm2d(out_channels)
        self.activation = activation(inplace=True)

        self.conv3 = Conv2dLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel=kernel,
            padding_mode=PaddingMode.SAME,
            apply_batchnorm=apply_batchnorm,
            apply_bias=apply_bias,
            activation=activation
        )

    def forward(self, x, skip_connection_map):
        x = self.conv1(x) + self.conv2(skip_connection_map)
        x = self.batchnorm2d(x)
        x = self.activation(x)
        return self.conv3(x) + x