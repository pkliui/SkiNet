from typing import Callable, Literal, cast

from torch import Tensor, nn

from SkiNet.ML.model.blocks.conv2d_layer import Conv2dLayer
from SkiNet.ML.utils.sampling.base_sampling import EncoderParams2D


class Merge2DBlock(nn.Module):
    """
    Merge2DBlock merges the output of a decoder with a skip connection.

    Instead of the standard UNet approach of concatenating decoder and skip features along the channel
    dimension followed by a single convolution, this block applies a separate convolution to each input
    and sums the results. This is algebraically equivalent to concatenation+convolution (by linearity)
    while keeping channel count constant and avoiding the peak memory cost of materialising the
    concatenated tensor.

    Three residual modes are supported for the post-merge refinement:

    ``local_refinement``:
        Post-activation pattern. BN and activation are applied to the merged sum, then a single
        conv refines it. The skip reuses the activated merged sum — not the raw sum:

            h = BN-Act(sum_of_convs)
            y = Conv-BN-Act(h) + h

    ``he1``:
        Pre-activation pattern with one refinement conv (He et al., ECCV 2016).
        BN and activation are applied before the conv; the skip is the raw merged sum:

            y = Conv(BN-Act(sum_of_convs)) + sum_of_convs

    ``he2``:
        Pre-activation pattern with two refinement convs (He et al., ECCV 2016).
        Same identity shortcut from the raw merged sum:

            h = Conv(BN-Act(sum_of_convs))
            y = Conv(BN-Act(h)) + sum_of_convs

    References:
        - He et al., "Identity Mappings in Deep Residual Networks" (ECCV 2016)

    :param layer_number: Position of this block within the decoder stack.
    :param in_channels_from_skip: Number of channels in the skip connection input to he merge block.
    :param in_channels_from_decoder: Number of channels in the decoder input to the merge block.
    :param out_channels: Number of output channels produced by the merge block.
    :param conv_params: Validated kernel, stride, dilation, and padding for the convolutional layers.
    :param residual_mode: Residual pattern — ``"local_refinement"``, ``"he1"``, or ``"he2"``. Defaults to ``"he2"``.
    :param activation: Factory for the non-linear activation (e.g. ``nn.ReLU``).
    """

    def __init__(self,
                 layer_number: int,
                 in_channels_from_skip: int,
                 in_channels_from_decoder: int,
                 out_channels: int,
                 conv_params: EncoderParams2D,
                 residual_mode: Literal["local_refinement", "he1", "he2"] = "he2",
                 activation: Callable[[], nn.Module] = nn.ReLU):
        super().__init__()
        self.layer_number = layer_number
        self.residual_mode = residual_mode

        self.merging_layer = True
        """Denotes if the layer merges the output of a decoder with a skip connection. Required for the forward method of the UNet."""

        # conv_x and conv_skip are always present: they project decoder and skip inputs to out_channels before summing
        self.conv_x = Conv2dLayer(in_channels=in_channels_from_decoder,
                                  out_channels=out_channels,
                                  kernel=conv_params.kernel,
                                  stride=(1, 1),
                                  dilation=conv_params.dilation,
                                  padding=conv_params.padding,
                                  apply_bias=True,
                                  apply_batchnorm=False,
                                  activation=None)

        self.conv_skip = Conv2dLayer(in_channels=in_channels_from_skip,
                                     out_channels=out_channels,
                                     kernel=conv_params.kernel,
                                     stride=(1, 1),
                                     dilation=conv_params.dilation,
                                     padding=conv_params.padding,
                                     apply_bias=True,
                                     apply_batchnorm=False,
                                     activation=None)

        if self.residual_mode == "local_refinement":
            self.activation = activation()
            self.batchnorm2d_out = nn.BatchNorm2d(out_channels)

            self.conv_refine = Conv2dLayer(in_channels=out_channels,
                                           out_channels=out_channels,
                                           kernel=conv_params.kernel,
                                           stride=(1, 1),
                                           dilation=conv_params.dilation,
                                           padding=conv_params.padding,
                                           apply_bias=True,
                                           apply_batchnorm=True,
                                           activation=activation)

        elif self.residual_mode == "he1":
            self.activation = activation()
            self.batchnorm2d_out = nn.BatchNorm2d(out_channels)

            self.conv_no_BNAct_refine = Conv2dLayer(in_channels=out_channels,
                                                    out_channels=out_channels,
                                                    kernel=conv_params.kernel,
                                                    stride=(1, 1),
                                                    dilation=conv_params.dilation,
                                                    padding=conv_params.padding,
                                                    apply_bias=True,
                                                    apply_batchnorm=False,
                                                    activation=None)

        elif self.residual_mode == "he2":
            self.activation1 = activation()
            self.batchnorm2d_out1 = nn.BatchNorm2d(out_channels)
            self.activation2 = activation()
            self.batchnorm2d_out2 = nn.BatchNorm2d(out_channels)

            self.conv_no_BNAct_refine1 = Conv2dLayer(in_channels=out_channels,
                                                     out_channels=out_channels,
                                                     kernel=conv_params.kernel,
                                                     stride=(1, 1),
                                                     dilation=conv_params.dilation,
                                                     padding=conv_params.padding,
                                                     apply_bias=True,
                                                     apply_batchnorm=False,
                                                     activation=None)

            self.conv_no_BNAct_refine2 = Conv2dLayer(in_channels=out_channels,
                                                     out_channels=out_channels,
                                                     kernel=conv_params.kernel,
                                                     stride=(1, 1),
                                                     dilation=conv_params.dilation,
                                                     padding=conv_params.padding,
                                                     apply_bias=True,
                                                     apply_batchnorm=False,
                                                     activation=None)

    def forward(self, x: Tensor, skip_connection_map: Tensor) -> Tensor:
        assert x.shape[0] == skip_connection_map.shape[0] and x.shape[2:] == skip_connection_map.shape[2:], \
            f"Batch size and spatial dims of decoder output {x.shape} and skip connection {skip_connection_map.shape} must match"
        sum_of_convs = self.conv_x(x) + self.conv_skip(skip_connection_map)

        if self.residual_mode == "he1":
            conv = self.conv_no_BNAct_refine(self.activation(self.batchnorm2d_out(sum_of_convs)))
            return cast(Tensor, conv + sum_of_convs)
        elif self.residual_mode == "he2":
            conv1 = self.conv_no_BNAct_refine1(self.activation1(self.batchnorm2d_out1(sum_of_convs)))
            conv2 = self.conv_no_BNAct_refine2(self.activation2(self.batchnorm2d_out2(conv1)))
            return cast(Tensor, conv2 + sum_of_convs)
        elif self.residual_mode == "local_refinement":
            h = self.activation(self.batchnorm2d_out(sum_of_convs))
            return cast(Tensor, self.conv_refine(h) + h)
        else:
            raise ValueError(f"Unknown residual_mode: {self.residual_mode!r}")
