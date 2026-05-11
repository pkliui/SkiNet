from typing import Callable, cast

from torch import Tensor, nn

from SkiNet.ML.model.blocks.conv2d_layer import Conv2dLayer
from SkiNet.ML.utils.sampling.base_sampling import EncoderParams2D


class AttentionGate(nn.Module):
    """
    Additive attention gate (Oktay et al., "Attention U-Net", MIDL 2018).

    Gates the skip-connection features using the decoder (gating) signal. Both inputs are
    projected to an intermediate space, summed, passed through ReLU + 1x1 conv + Sigmoid to
    produce a spatial attention map, and the map is multiplied onto the skip features.

    Bias in the 1x1 conv is disabled since the batchnorm provides a learnable shift.

    :param F_g: Channels in the gating signal (decoder output).
    :param F_l: Channels in the skip-connection features.
    :param F_int: Intermediate channel dimension (used only internally to compute the attention map alpha).
        Typically F_l // 2.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(F_int))  # (B, F_int, H, W)
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
                                 nn.BatchNorm2d(F_int))  # (B, F_int, H, W)
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1, bias=False),  # (B, 1, H, W)
                                 nn.BatchNorm2d(1),
                                 nn.Sigmoid())  # scale vector to [0, 1]
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: Tensor, x: Tensor) -> Tensor:
        """
        Compute alpha, per-spatial-location scalar attention map, and apply it to the skip features.

        :param g: Gating signal from decoder — shape (B, F_g, H, W).
        :param x: Skip-connection features — shape (B, F_l, H, W).
        :returns: Gated skip features — shape (B, F_l, H, W).
        """
        alpha = cast(Tensor, self.psi(self.relu(self.W_g(g) + self.W_x(x))))  # (B, 1, H, W)
        return x * alpha  # (B, F_l, H, W)


class LocalRefinementMerge(nn.Module):
    """
    Post-activation merge refinement (Oktay et al. Evaluation of Deep Learning
    to Augment Image-Guided Radiotherapy for Head and Neck and Prostate Cancers, JAMA Network Open 2020)

        h = BN-Act(sum_of_convs)
        y = Conv-BN-Act(h) + h
    """

    def __init__(self,
                 in_channels_from_skip: int,
                 in_channels_from_decoder: int,
                 out_channels: int,
                 conv_params: EncoderParams2D,
                 activation: Callable[[], nn.Module]):
        super().__init__()
        self.conv_x, self.conv_skip = _projection_pair(
            in_channels_from_decoder, in_channels_from_skip, out_channels, conv_params)

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

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        h = self.activation(self.batchnorm2d_out(self.conv_x(x) + self.conv_skip(skip)))
        return cast(Tensor, self.conv_refine(h) + h)


class He1Merge(nn.Module):
    """
    Pre-activation merge refinement with one conv (He et al., ECCV 2016).

        y = Conv(BN-Act(sum_of_convs)) + sum_of_convs
    """

    def __init__(self,
                 in_channels_from_skip: int,
                 in_channels_from_decoder: int,
                 out_channels: int,
                 conv_params: EncoderParams2D,
                 activation: Callable[[], nn.Module]):
        super().__init__()
        self.conv_x, self.conv_skip = _projection_pair(
            in_channels_from_decoder, in_channels_from_skip, out_channels, conv_params)

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

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        merged = self.conv_x(x) + self.conv_skip(skip)
        conv = self.conv_no_BNAct_refine(self.activation(self.batchnorm2d_out(merged)))
        return cast(Tensor, conv + merged)


class He2Merge(nn.Module):
    """
    Pre-activation merge refinement with two convs (He et al., ECCV 2016).

        h = Conv(BN-Act(sum_of_convs))
        y = Conv(BN-Act(h)) + sum_of_convs
    """

    def __init__(self,
                 in_channels_from_skip: int,
                 in_channels_from_decoder: int,
                 out_channels: int,
                 conv_params: EncoderParams2D,
                 activation: Callable[[], nn.Module]):
        super().__init__()
        self.conv_x, self.conv_skip = _projection_pair(
            in_channels_from_decoder, in_channels_from_skip, out_channels, conv_params)

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

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        merged = self.conv_x(x) + self.conv_skip(skip)
        conv1 = self.conv_no_BNAct_refine1(self.activation1(self.batchnorm2d_out1(merged)))
        conv2 = self.conv_no_BNAct_refine2(self.activation2(self.batchnorm2d_out2(conv1)))
        return cast(Tensor, conv2 + merged)


class AttentionGateMerge(nn.Module):
    """
    Attention-gated merge (Oktay et al., MIDL 2018) with he2 post-merge refinement.

    The decoder features gate the skip connection via an additive attention gate before
    the projection-and-sum merge. Post-merge refinement follows the he2 pre-activation
    pattern with identity shortcut:

        x_skip_attended = AG(decoder, skip)
        merged          = conv_x(decoder) + conv_skip(x_skip_attended)
        h               = Conv(BN-Act(merged))
        y               = Conv(BN-Act(h)) + merged
    """

    def __init__(self,
                 in_channels_from_skip: int,
                 in_channels_from_decoder: int,
                 out_channels: int,
                 conv_params: EncoderParams2D,
                 activation: Callable[[], nn.Module]):
        super().__init__()
        self.attention_gate = AttentionGate(F_g=in_channels_from_decoder,
                                            F_l=in_channels_from_skip,
                                            F_int=max(out_channels // 2, 1))

        self.conv_x, self.conv_skip = _projection_pair(
            in_channels_from_decoder, in_channels_from_skip, out_channels, conv_params)

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

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        attended_skip = self.attention_gate(x, skip)  # (B, F_l, H, W)
        merged = self.conv_x(x) + self.conv_skip(attended_skip)  # (B, out_channels, H, W)
        conv1 = self.conv_no_BNAct_refine1(self.activation1(self.batchnorm2d_out1(merged)))  # (B, out_channels, H, W)
        conv2 = self.conv_no_BNAct_refine2(self.activation2(self.batchnorm2d_out2(conv1)))  # (B, out_channels, H, W)
        return cast(Tensor, conv2 + merged)  # (B, out_channels, H, W)


def _projection_pair(in_channels_from_decoder: int,
                     in_channels_from_skip: int,
                     out_channels: int,
                     conv_params: EncoderParams2D) -> tuple[nn.Module, nn.Module]:
    """Shared conv_x / conv_skip projection layers used by every merge mode."""
    conv_x = Conv2dLayer(in_channels=in_channels_from_decoder,
                         out_channels=out_channels,
                         kernel=conv_params.kernel,
                         stride=(1, 1),
                         dilation=conv_params.dilation,
                         padding=conv_params.padding,
                         apply_bias=True,
                         apply_batchnorm=False,
                         activation=None)

    conv_skip = Conv2dLayer(in_channels=in_channels_from_skip,
                            out_channels=out_channels,
                            kernel=conv_params.kernel,
                            stride=(1, 1),
                            dilation=conv_params.dilation,
                            padding=conv_params.padding,
                            apply_bias=True,
                            apply_batchnorm=False,
                            activation=None)
    return conv_x, conv_skip
