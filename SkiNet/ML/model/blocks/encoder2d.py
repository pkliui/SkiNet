import torch
import torch.nn as nn
from typing import Optional, Callable, Union, Iterable

from SkiNet.ML.model.blocks.conv2d_layer import Conv2dLayer
from SkiNet.ML.utils.sampling.encoder_sampling import PaddingMode

class Encoder2D(nn.Module):
    """
        Encoder2D is an encoder block

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel: Kernel size of the convolution operation.
        :param padding_mode: Padding mode applied to input. Should be of type PaddingMode.
            Possible values are 'SAME' (padding applied to keep same spatial dimensions),
            and 'DOWNSAMPLING_FACTOR_2' (downsample the output by factor 2).
            Default is 'SAME'.
        :param dilation: Dilation factor of the convolution operation.
            Default is 1
        :param apply_bias: If True, applies a learnable bias to the convolution output. Ignored if apply_batchnorm=True.
            Default is False
        :param apply_batchnorm: If True, applies batch normalization after convolution. When enabled, the `apply_bias` parameter
            is ignored and set to False.
            Default is True
        :param activation: Non-linear activation function applied after batch normalization. If None, no activation is applied.
            Default is torch.nn.ReLU
        :param use_residual: If True, adds a residual connection from input to output. Default is True.
        :param layer_number: The layer number within the encoder block. The upper-most layer is 0. Default is 0.

        :return: Output tensor after applying convolution, optional batch normalization, and optional activation.
        """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: Union[int, Iterable[int]] = 3,
                 padding_mode: PaddingMode = PaddingMode.SAME,
                 dilation: Union[int, Iterable[int]] = 1,
                 apply_bias: bool = False,
                 apply_batchnorm: bool = True,
                 activation: Optional[Callable] = torch.nn.ReLU,
                 use_residual: bool = True,
                 layer_number=0):
        super().__init__()

        self.use_residual = use_residual
        self.layer_number = layer_number

        self.conv2d_layer1 = Conv2dLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel=kernel,
                    padding_mode=padding_mode,
                    dilation=dilation,
                    apply_bias=apply_bias,
                    apply_batchnorm=apply_batchnorm,
                    activation=activation)

        self.conv2d_layer2 = Conv2dLayer(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel=kernel,
                    padding_mode=PaddingMode.SAME, # second conv always uses SAME padding to preserve dimensions
                    dilation=dilation,
                    apply_bias=apply_bias,
                    apply_batchnorm=apply_batchnorm,
                    activation=activation)

    def forward(self, x):
        x = self.conv2d_layer1(x)
        conv2 = self.conv2d_layer2(x)
        return conv2 + x if self.use_residual else conv2