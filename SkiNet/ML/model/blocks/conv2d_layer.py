from typing import Callable, Optional, Tuple, cast

import torch.nn as nn
from torch import Tensor

from SkiNet.ML.utils.sampling.encoder_sampling import ConvParams, get_padding
from SkiNet.ML.utils.typing_utils import IntOrTuple2d


class Conv2dLayer(nn.Module):
    """Conv2dLayer is a 2D convolution layer followed by optional batch normalization and activation.

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

    :return: Output tensor after applying convolution, optional batch normalization, and optional activation.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: IntOrTuple2d,
                 stride: IntOrTuple2d,
                 dilation: IntOrTuple2d,
                 apply_bias: bool = False,
                 apply_batchnorm: bool = True,
                 activation: Optional[Callable] = nn.ReLU):
        super().__init__()

        params = ConvParams.from_inputs(kernel=kernel,
                                        dilation=dilation,
                                        stride=stride,
                                        num_dims=2)
        # Narrow the type for Conv2d
        self.padding: Tuple[int, int] = cast(tuple[int, int], get_padding(params))
        """Padding value calculated based on kernel size and dilation"""

        self.apply_batchnorm = apply_batchnorm
        if self.apply_batchnorm:
            self.apply_bias = False
        else:
            self.apply_bias = apply_bias

        self.conv2d = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel,
                                stride=stride,
                                padding=self.padding,
                                dilation=dilation,
                                bias=self.apply_bias)
        """Conv 2d layer"""

        self.batchnorm2d = nn.BatchNorm2d(out_channels) if self.apply_batchnorm else None
        """Batchnorm layer"""

        self.activation = activation(inplace=True) if activation else None
        """Non-linear activation layer"""

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv2d(x)
        if self.batchnorm2d is not None:
            x = self.batchnorm2d(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
