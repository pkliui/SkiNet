from typing import Callable, cast

import torch.nn as nn
from torch import Tensor
from torchvision.ops import SqueezeExcitation

from SkiNet.ML.model.blocks.conv2d_layer import Conv2dLayer
from SkiNet.ML.utils.sampling.base_sampling import EncoderParams2D


class LocalRefinementEncoder(nn.Module):
    """
    Post-activation residual encoder (Oktay et al. Evaluation of Deep Learning
    to Augment Image-Guided Radiotherapy for Head and Neck and Prostate Cancers, JAMA Network Open 2020)

        h = Conv-BN-Act(x)      # downsamples
        y = Conv-BN-Act(h) + h  # refines; skip from downsampled intermediate h, not an identity shortcut

    :param in_channels: Number of input channels into the encoder block.
    :param out_channels: Number of output channels out of the encoder block.
    :param conv_params: Convolutional parameters for the encoder block.
    :param apply_bias: Whether to apply bias in the convolutional layers.
    :param activation: Activation function to use in the convolutional layers.
    :param use_residual: Whether to use the residual connection (skip from downsampled intermediate h).
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_params: EncoderParams2D,
                 apply_bias: bool,
                 activation: Callable[[], nn.Module],
                 use_residual: bool):
        super().__init__()
        self.use_residual = use_residual

        self.conv_downsample = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=conv_params.kernel,
            stride=conv_params.stride,
            dilation=conv_params.dilation,
            padding=conv_params.padding,
            apply_bias=apply_bias,
            apply_batchnorm=True,
            activation=activation)

        self.conv_refine = Conv2dLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel=conv_params.kernel,
            stride=(1, 1),
            dilation=conv_params.dilation,
            padding=conv_params.padding,
            apply_bias=apply_bias,
            apply_batchnorm=True,
            activation=activation)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv_downsample(x)
        conv2 = self.conv_refine(h)
        return cast(Tensor, conv2 + h if self.use_residual else conv2)


class He2Encoder(nn.Module):
    """
    Pre-activation residual encoder (He et al. Identity Mappings in Deep Residual Networks, ECCV 2016).

        h = Conv(BN-Act(x))         # downsamples
        y = Conv(BN-Act(h)) + P(x)  # refines; P is a 1×1 projection shortcut

    :param in_channels: Number of input channels into the encoder block.
    :param out_channels: Number of output channels out of the encoder block.
    :param conv_params: Convolutional parameters for the encoder block.
    :param apply_bias: Whether to apply bias in the convolutional layers.
    :param activation: Activation function to use in the convolutional layers.
    :param use_residual: Whether to use the residual connection (projection shortcut P).
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_params: EncoderParams2D,
                 apply_bias: bool,
                 activation: Callable[[], nn.Module],
                 use_residual: bool):
        super().__init__()
        if not use_residual:
            raise ValueError("use_residual=False is incompatible with he2")

        self.batchnorm2d_in = nn.BatchNorm2d(in_channels)
        self.batchnorm2d_out = nn.BatchNorm2d(out_channels)
        self.activation = activation()

        self.conv_no_BNAct_downsample = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=conv_params.kernel,
            stride=conv_params.stride,
            dilation=conv_params.dilation,
            padding=conv_params.padding,
            apply_bias=apply_bias,
            apply_batchnorm=False,
            activation=None)

        self.conv_no_BNAct_refine = Conv2dLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel=conv_params.kernel,
            stride=(1, 1),
            dilation=conv_params.dilation,
            padding=conv_params.padding,
            apply_bias=apply_bias,
            apply_batchnorm=False,
            activation=None)

        self.shortcut = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=(1, 1),
            stride=conv_params.stride,
            dilation=(1, 1),
            padding=(0, 0),
            apply_bias=apply_bias,
            apply_batchnorm=False,
            activation=None)

    def forward(self, x: Tensor) -> Tensor:
        conv1 = self.conv_no_BNAct_downsample(self.activation(self.batchnorm2d_in(x)))
        conv2 = self.conv_no_BNAct_refine(self.activation(self.batchnorm2d_out(conv1)))
        return cast(Tensor, conv2 + self.shortcut(x))


class ClassicalEncoder(nn.Module):
    """
    Post-activation encoder from the original UNet (Ronneberger et al., MICCAI 2015).

    Two sequential Conv-BN-Act blocks with no residual connection:

        h = Conv-BN-Act(x)   # downsamples
        y = Conv-BN-Act(h)   # refines

    ``use_residual`` is accepted for interface compatibility with the encoder registry
    but ignored — the classical UNet encoder has no skip connection.

    :param in_channels: Number of input channels into the encoder block.
    :param out_channels: Number of output channels out of the encoder block.
    :param conv_params: Convolutional parameters for the encoder block.
    :param apply_bias: Whether to apply bias in the convolutional layers.
    :param activation: Activation function to use in the convolutional layers.
    :param use_residual: Ignored. Accepted for registry compatibility only.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_params: EncoderParams2D,
                 apply_bias: bool,
                 activation: Callable[[], nn.Module],
                 use_residual: bool):
        super().__init__()

        self.conv_downsample = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=conv_params.kernel,
            stride=conv_params.stride,
            dilation=conv_params.dilation,
            padding=conv_params.padding,
            apply_bias=apply_bias,
            apply_batchnorm=True,
            activation=activation)

        self.conv_refine = Conv2dLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel=conv_params.kernel,
            stride=(1, 1),
            dilation=conv_params.dilation,
            padding=conv_params.padding,
            apply_bias=apply_bias,
            apply_batchnorm=True,
            activation=activation)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.conv_refine(self.conv_downsample(x)))


class SEEncoder(nn.Module):
    """
    Pre-activation residual encoder with Squeeze-and-Excitation channel attention
    (He et al., ECCV 2016 + Hu et al., CVPR 2018).

    Same skeleton as He2Encoder; SE is applied to the refinement output before the
    projection shortcut addition, recalibrating the conv path only:

        h     = Conv(BN-Act(x))      # downsamples
        conv2 = Conv(BN-Act(h))      # refines
        y     = SE(conv2) + P(x)     # SE-recalibrated output + 1×1 projection shortcut

    :param in_channels: Number of input channels into the encoder block.
    :param out_channels: Number of output channels out of the encoder block.
    :param conv_params: Convolutional parameters for the encoder block.
    :param apply_bias: Whether to apply bias in the convolutional layers.
    :param activation: Activation function to use in the convolutional layers.
    :param use_residual: Whether to use the residual connection (projection shortcut P).
    :param se_reduction: Reduction ratio for the Squeeze-and-Excitation block,
        controlling the bottleneck in the channel attention mechanism.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_params: EncoderParams2D,
                 apply_bias: bool,
                 activation: Callable[[], nn.Module],
                 use_residual: bool,
                 se_reduction: int = 16):
        super().__init__()
        if not use_residual:
            raise ValueError("use_residual=False is incompatible with se")

        self.batchnorm2d_in = nn.BatchNorm2d(in_channels)
        self.batchnorm2d_out = nn.BatchNorm2d(out_channels)
        self.activation = activation()

        self.conv_no_BNAct_downsample = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=conv_params.kernel,
            stride=conv_params.stride,
            dilation=conv_params.dilation,
            padding=conv_params.padding,
            apply_bias=apply_bias,
            apply_batchnorm=False,
            activation=None)

        self.conv_no_BNAct_refine = Conv2dLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel=conv_params.kernel,
            stride=(1, 1),
            dilation=conv_params.dilation,
            padding=conv_params.padding,
            apply_bias=apply_bias,
            apply_batchnorm=False,
            activation=None)

        squeeze_channels = max(out_channels // se_reduction, 4)
        self.channel_attention = SqueezeExcitation(input_channels=out_channels,
                                                   squeeze_channels=squeeze_channels)

        self.shortcut = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=(1, 1),
            stride=conv_params.stride,
            dilation=(1, 1),
            padding=(0, 0),
            apply_bias=apply_bias,
            apply_batchnorm=False,
            activation=None)

    def forward(self, x: Tensor) -> Tensor:
        conv1 = self.conv_no_BNAct_downsample(self.activation(self.batchnorm2d_in(x)))
        conv2 = self.conv_no_BNAct_refine(self.activation(self.batchnorm2d_out(conv1)))
        return cast(Tensor, self.channel_attention(conv2) + self.shortcut(x))
