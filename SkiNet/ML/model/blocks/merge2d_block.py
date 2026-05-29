from typing import Callable, Literal, cast

from torch import Tensor, nn

from SkiNet.ML.model.blocks.merge2d_residual_blocks import (AttentionGateMerge, ClassicalMerge, He1Merge, He2Merge,
                                                            LocalRefinementMerge)
from SkiNet.ML.utils.sampling.base_sampling import EncoderParams2D

_MERGE_REGISTRY: dict[str, type[nn.Module]] = {
    "classical": ClassicalMerge,
    "local_refinement": LocalRefinementMerge,
    "he1": He1Merge,
    "he2": He2Merge,
    "attention_gate": AttentionGateMerge,
}


class Merge2DBlock(nn.Module):
    """
    Merges the output of a decoder with a skip connection.

    Instead of the standard UNet approach of concatenating decoder and skip features along the
    channel dimension followed by a single convolution, this block applies a separate convolution
    to each input and sums the results. This is algebraically equivalent to concatenation+convolution
    (by linearity) while keeping channel count constant and avoiding the peak memory cost of
    materialising the concatenated tensor.

    Delegates post-merge refinement to a mode-specific block; see the corresponding class in
    ``merge2d_residual_blocks`` for the full forward-pass description of each mode.

    ``classical``:
        Concatenation-based merge from the original UNet (Ronneberger et al., MICCAI 2015).
        Concatenates decoder and skip features, then applies two Conv-BN-Act blocks with no residual.
        See :class:`ClassicalMerge`.

    ``local_refinement``:
        Post-activation refinement with skip from activated intermediate.
        See :class:`LocalRefinementMerge`.

    ``he1``:
        Pre-activation single-conv refinement with identity shortcut (He et al., ECCV 2016).
        See :class:`He1Merge`.

    ``he2``:
        Pre-activation double-conv refinement with identity shortcut (He et al., ECCV 2016).
        See :class:`He2Merge`.

    ``attention_gate``:
        Additive attention gate on the skip connection (Oktay et al., MIDL 2018) followed
        by he2-style post-merge refinement.
        See :class:`AttentionGateMerge`.

    :param layer_number: Position of this block within the decoder stack.
    :param in_channels_from_skip: Number of channels arriving from the skip connection.
    :param in_channels_from_decoder: Number of channels arriving from the decoder.
    :param out_channels: Number of output channels produced by the merge block, it is normally the number of channels
        from decoder and from the skip connection, which are the same.
    :param conv_params: Validated kernel, stride, dilation, and padding for the convolutional layers.
    :param residual_mode: One of ``"local_refinement"``, ``"he1"``, ``"he2"``, ``"attention_gate"``. Defaults to ``"he2"``.
    :param activation: Factory for the non-linear activation (e.g. ``nn.ReLU``).
    """

    def __init__(self,
                 layer_number: int,
                 in_channels_from_skip: int,
                 in_channels_from_decoder: int,
                 out_channels: int,
                 conv_params: EncoderParams2D,
                 residual_mode: Literal["classical", "local_refinement", "he1", "he2", "attention_gate"] = "he2",
                 activation: Callable[[], nn.Module] = nn.ReLU):
        super().__init__()
        self.layer_number = layer_number
        self.residual_mode = residual_mode

        self.merging_layer = True
        """Denotes if the layer merges the output of a decoder with a skip connection. Required for the forward method of the UNet."""

        if residual_mode not in _MERGE_REGISTRY:
            raise ValueError(
                f"Unknown residual_mode: {residual_mode!r}. Choose from {list(_MERGE_REGISTRY)}")

        self._block = _MERGE_REGISTRY[residual_mode](
            in_channels_from_skip=in_channels_from_skip,
            in_channels_from_decoder=in_channels_from_decoder,
            out_channels=out_channels,
            conv_params=conv_params,
            activation=activation,
        )

    def forward(self, x: Tensor, skip_connection_map: Tensor) -> Tensor:
        assert x.shape[0] == skip_connection_map.shape[0] and x.shape[2:] == skip_connection_map.shape[2:], \
            f"Batch size and spatial dims of decoder output {x.shape} and skip connection {skip_connection_map.shape} must match"
        return cast(Tensor, self._block(x, skip_connection_map))
