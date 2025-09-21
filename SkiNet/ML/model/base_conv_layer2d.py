import torch
from typing import Optional, Callable, Union, Iterable
from SkiNet.ML.utils.sampling.encoder_sampling import PaddingMode, get_padding_value


class BaseConv2D(torch.nn.Module):
    """
    BaseConv2D is a 2D convolution block followed by optional batch normalization and activation.

    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param kernel: Kernel size of the convolution operation.
    :param stride: Stride of the convolution operation
    :param padding_mode: Padding mode applied to input. Should be of type PaddingMode.
        Possible values are 'VALID' (no padding applied), 'SAME' (padding applied to keep same spatial dimensions), 
        and 'DOWNSAMPLING_FACTOR_2' (downsample the output by factor 2).

    :param dilation: Dilation factor of the convolution operation.
    :param bias: If True, adds a learnable bias to the output.
    :param apply_batchnorm: If True, applies batch normalization after convolution.
    :param activation: Non-linear activation function applied after batch normalization. If None, no activation is applied.
    
    :return: Output tensor after applying convolution, optional batch normalization, and optional activation.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels:  int,
                 kernel: Union[int, Iterable[int]],
                 stride: Union[int, Iterable[int]] = 1,
                 padding_mode:  PaddingMode = PaddingMode.SAME,
                 dilation: Union[int, Iterable[int]] = 1,
                 bias: bool = False,
                 apply_batchnorm: bool = True,
                 activation: Optional[Callable] = torch.nn.ReLU
                 ):
        super().__init__()

        if not isinstance(padding_mode, PaddingMode):
            raise ValueError(f"Invalid padding mode: {padding_mode}. Must be one of {list(PaddingMode)}.")
        self.padding = get_padding_value(stride=stride,
                                                 kernel=kernel,
                                                 dilation=dilation,
                                                 padding_mode=padding_mode,
                                                 num_dims=2)
        """Padding value calculated based on padding mode, kernel size, stride and dilation"""
        
        self.conv2d = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=self.padding,
            dilation=dilation,
            bias=bias
        )
        """Conv 2d layer"""
       
        self.apply_batchnorm = apply_batchnorm
        if self.apply_batchnorm:
            self.bias = False
            self.batchnorm2d = torch.nn.BatchNorm2d(out_channels)
        else:
            self.bias = True
            self.batchnorm2d = None
        """Bias is not used if batchnorm is applied. This is because batchnorm has its own learnable parameters that can offset the mean of the activations."""
        """Batchnorm layer"""

        self.activation = activation(inplace=True) if activation else None
        """Non-linear activation layer"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2d(x)
        if self.batchnorm2d is not None:
            x = self.batchnorm2d(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
