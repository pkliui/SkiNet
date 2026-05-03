from typing import Callable, Literal, cast

import torch.nn as nn
from torch import Tensor

from SkiNet.ML.model.blocks.conv2d_layer import Conv2dLayer
from SkiNet.ML.utils.sampling.base_sampling import EncoderParams2D


class Encoder2D(nn.Module):
    """
    Encoder2D is an encoder block composed of a downsampling and a shape-preserving convolutional layer.

    Two residual modes are supported:

    ``local_refinement``:
        Post-activation pattern. The first conv downsamples, the second refines at the downsampled
        resolution, and the skip connection reuses the downsampled intermediate — not the original
        block input:

            h = Conv-BN-Act(x)          # downsamples
            y = Conv-BN-Act(h) + h      # refines; skip from h, not x

        Because x and y have mismatched spatial dimensions, a true identity shortcut from x is not
        possible. Using h as the skip preserves a direct additive path through the downsampled features.

    ``he2``:
        Pre-activation pattern (He et al., "Identity Mappings in Deep Residual Networks", ECCV 2016).
        BN and activation are applied before each conv, keeping the residual branch as a clean identity:

            h = Conv(BN-Act(x))         # downsamples; no BN/Act inside conv
            y = Conv(BN-Act(h)) + P(x)  # refines; P is a 1×1 projection shortcut

        Requires ``use_residual=True`` — enforced at construction.

    References:
        - He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)
        - He et al., "Identity Mappings in Deep Residual Networks" (ECCV 2016)

    :param layer_number: Position of this block within the encoder stack; 1 is the topmost layer.
    :param in_channels: Number of input channels into the encoder block.
    :param out_channels: Number of output channels out of the encoder block.
    :param conv_params: Validated kernel, stride, dilation, and padding for the convolutional layers.
    :param apply_bias: If True, adds a learnable bias. Ignored (forced False) when ``apply_batchnorm=True``
        at Conv2dLayer construction because BN subsumes the bias.
    :param activation: Factory for the non-linear activation (e.g. ``nn.ReLU``).
    :param use_residual: If True, adds a skip connection. Must be True for ``he2`` and if not, raises an error.
    :param residual_mode: Residual pattern — ``"local_refinement"`` or ``"he2"``. Defaults to ``"he2"``.
    """

    def __init__(self,
                 layer_number: int,
                 in_channels: int,
                 out_channels: int,
                 conv_params: EncoderParams2D,
                 apply_bias: bool,
                 activation: Callable[[], nn.Module],
                 use_residual: bool,
                 residual_mode: Literal["local_refinement",
                                        "he2"] = "he2"):
        super().__init__()
        self.layer_number = layer_number
        self.out_channels = out_channels
        self.use_residual = use_residual
        self.residual_mode = residual_mode

        self.merging_layer = False
        """Denotes if the layer merges the output of a decoder with a skip connection. Required for the forward method of the UNet."""

        if self.residual_mode == "local_refinement":
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

            # stride=1 keeps the spatial size of the downsampled output
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

        elif self.residual_mode == "he2":
            if not use_residual:
                raise ValueError("use_residual=False is incompatible with he2 and is ignored")

            # batchnorm2d_in normalises in_channels features; batchnorm2d_out normalises out_channels features
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

            # stride=1 keeps the spatial size of the downsampled output
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

            # 1×1 projection shortcut to match spatial and channel dimensions of the block output
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
        if self.residual_mode == "local_refinement":
            conv1 = self.conv_downsample(x)
            conv2 = self.conv_refine(conv1)
            # skip from the downsampled intermediate, not the original block input
            return cast(Tensor, conv2 + conv1 if self.use_residual else conv2)
        elif self.residual_mode == "he2":
            conv1 = self.conv_no_BNAct_downsample(self.activation(self.batchnorm2d_in(x)))
            conv2 = self.conv_no_BNAct_refine(self.activation(self.batchnorm2d_out(conv1)))
            return cast(Tensor, conv2 + self.shortcut(x))
        else:
            raise ValueError(f"Unknown residual_mode: {self.residual_mode!r}")
