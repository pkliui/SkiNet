from typing import Callable, Optional, cast

import torch.nn as nn
from torch import Tensor

from SkiNet.ML.utils.typing_utils import TupleOfInt2d


class Conv2dLayer(nn.Module):
    """
    Conv2dLayer is a 2D convolution layer followed by optional batch normalization and activation.

    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param kernel: Kernel size of the convolution operation.
    :param stride: Stride of the convolution operation.
    :param dilation: Dilation factor of the convolution operation.
    :param padding: The amount of padding to be applied to the input from each side. The padding mode is kept to be nn.conv2d's default "zeros"
    :param apply_bias: If True, adds a learnable bias to the output. Note that this parameter is ignored and set to False
        when `apply_batchnorm=True` because batch normalization includes its own learnable parameters.
    :param apply_batchnorm: If True, applies batch normalization after convolution.
    :param activation: Non-linear activation function applied after batch normalization.
        Can be provided either as a module class (e.g. default, nn.ReLU) or as a callable returning
        an nn.Module (e.g. lambda: nn.ReLU(inplace=True)). If None, no activation is applied.
    :return: Output tensor after applying convolution, optional batch normalization, and optional activation.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: TupleOfInt2d,
                 stride: TupleOfInt2d,
                 dilation: TupleOfInt2d,
                 padding: TupleOfInt2d,
                 apply_bias: bool,
                 apply_batchnorm: bool,
                 activation: Optional[Callable[[], nn.Module]] = nn.ReLU):
        super().__init__()

        if apply_batchnorm:
            self.apply_bias = False
        else:
            self.apply_bias = apply_bias

        self.conv2d = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                bias=self.apply_bias)

        self.batchnorm2d = nn.BatchNorm2d(out_channels) if apply_batchnorm else None
        self.activation = activation() if activation else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv2d(x)
        if self.batchnorm2d is not None:
            x = self.batchnorm2d(x)
        if self.activation is not None:
            # cast for mypy
            x = cast(Tensor, self.activation(x))
        return x
