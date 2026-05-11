from typing import Callable, Literal, cast

import torch.nn as nn
from torch import Tensor

from SkiNet.ML.model.blocks.encoder2d_residual_blocks import He2Encoder, LocalRefinementEncoder, SEEncoder
from SkiNet.ML.utils.sampling.base_sampling import EncoderParams2D

_ENCODER_REGISTRY: dict[str, type[nn.Module]] = {
    "local_refinement": LocalRefinementEncoder,
    "he2": He2Encoder,
    "se": SEEncoder,
}


class Encoder2D(nn.Module):
    """
    Encoder block composed of a downsampling and a shape-preserving convolutional layer.

    Delegates to a mode-specific residual block; see the corresponding class in
    ``encoder2d_residual_blocks`` for the full forward-pass description of each mode.

    ``local_refinement``:
        Post-activation pattern. Skip reuses the downsampled intermediate ``h``.
        See :class:`LocalRefinementEncoder`.

    ``he2``:
        Pre-activation pattern (He et al., ECCV 2016) with 1×1 projection shortcut.
        See :class:`He2Encoder`.

    ``se``:
        Pre-activation skeleton (he2) with Squeeze-and-Excitation channel attention
        applied to the refinement output before the shortcut addition (Hu et al., CVPR 2018).
        See :class:`SEEncoder`.

    :param layer_number: Position of this block within the encoder stack; 1 is the topmost layer.
    :param in_channels: Number of input channels into the encoder block.
    :param out_channels: Number of output channels out of the encoder block.
    :param conv_params: Validated kernel, stride, dilation, and padding for the convolutional layers.
    :param apply_bias: If True, adds a learnable bias.
    :param activation: Factory for the non-linear activation (e.g. ``nn.ReLU``).
    :param use_residual: If True, adds a skip connection. Must be True for ``he2`` and ``se``.
    :param residual_mode: One of ``"local_refinement"``, ``"he2"``, ``"se"``. Defaults to ``"local_refinement"``.
    :param se_reduction: Reduction factor for the Squeeze-and-Excitation block.
        Only used if ``residual_mode`` is ``"se"``. Defaults to 16.
    """

    def __init__(self,
                 layer_number: int,
                 in_channels: int,
                 out_channels: int,
                 conv_params: EncoderParams2D,
                 apply_bias: bool,
                 activation: Callable[[], nn.Module],
                 use_residual: bool,
                 residual_mode: Literal["local_refinement", "he2", "se"] = "local_refinement",
                 se_reduction: int = 16):
        super().__init__()
        self.layer_number = layer_number
        self.out_channels = out_channels
        self.use_residual = use_residual
        self.residual_mode = residual_mode

        self.merging_layer = False
        """Denotes if the layer merges the output of a decoder with a skip connection.
        Required for the forward method of the UNet."""

        if residual_mode not in _ENCODER_REGISTRY:
            raise ValueError(
                f"Unknown residual_mode: {residual_mode!r}. Choose from {list(_ENCODER_REGISTRY)}")

        extra_kwargs = {"se_reduction": se_reduction} if residual_mode == "se" else {}
        self._block = _ENCODER_REGISTRY[residual_mode](
            in_channels=in_channels,
            out_channels=out_channels,
            conv_params=conv_params,
            apply_bias=apply_bias,
            activation=activation,
            use_residual=use_residual,
            **extra_kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self._block(x))
