import pytest
import torch
from torch import isfinite, randn, sqrt
from torch.nn import MSELoss, ReLU

from SkiNet.ML.model.blocks.decoder2d import Decoder2D
from SkiNet.ML.model.blocks.merge2d_block import Merge2DBlock
from SkiNet.ML.model.blocks.merge2d_residual_blocks import (AttentionGateMerge, He1Merge, He2Merge,
                                                            LocalRefinementMerge)
from SkiNet.ML.utils.sampling.decoder_sampling import get_decoder_params_2d
from SkiNet.ML.utils.sampling.encoder_sampling import get_encoder_params_2d

_ALL_MODES = ["local_refinement", "he1", "he2", "attention_gate"]


def _inner_kwargs(mode: str, in_channels_from_skip: int = 4, in_channels_from_decoder: int = 4,
                  out_channels: int = 4) -> dict:
    """Shared kwargs for constructing residual block subclasses directly in structural tests."""
    return dict(
        in_channels_from_skip=in_channels_from_skip,
        in_channels_from_decoder=in_channels_from_decoder,
        out_channels=out_channels,
        conv_params=get_encoder_params_2d(kernel=(3, 3), stride=(1, 1), dilation=(1, 1)),
        activation=ReLU,
    )


def _make_merge(residual_mode: str,
                in_channels_from_skip: int = 4,
                in_channels_from_decoder: int = 4,
                out_channels: int = 4,
                kernel: tuple[int, int] = (3, 3),
                stride: tuple[int, int] = (2, 2),
                dilation: tuple[int, int] = (1, 1)) -> Merge2DBlock:
    params = get_encoder_params_2d(kernel=kernel, stride=stride, dilation=dilation)
    return Merge2DBlock(layer_number=0,
                        in_channels_from_skip=in_channels_from_skip,
                        in_channels_from_decoder=in_channels_from_decoder,
                        out_channels=out_channels,
                        conv_params=params,
                        residual_mode=residual_mode,  # type: ignore[arg-type]
                        activation=ReLU)


# ---------------------------------------------------------------------------
# Output shape: all four modes must be identical
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("residual_mode", _ALL_MODES)
@pytest.mark.parametrize("batch_size, in_channels, out_channels, input_shape, kernel, stride, dilation", [
    (2, 8, 4, (32, 32), (3, 3), (2, 2), (1, 1)),
])
def test_merge2d_block_output_shape(batch_size: int, in_channels: int, out_channels: int,
                                    input_shape: tuple[int, int], kernel: tuple[int, int],
                                    stride: tuple[int, int], dilation: tuple[int, int],
                                    residual_mode: str) -> None:
    """Output shape matches (batch, out_channels, H, W) for all four residual modes."""
    params = get_encoder_params_2d(kernel=kernel, stride=stride, dilation=dilation)
    decoder = Decoder2D(in_channels=in_channels,
                        out_channels=out_channels,
                        decoder_params=get_decoder_params_2d(params),
                        activation=ReLU,
                        layer_number=1)
    merge_block = Merge2DBlock(layer_number=0,
                               in_channels_from_skip=out_channels,
                               in_channels_from_decoder=out_channels,
                               out_channels=out_channels,
                               conv_params=params,
                               residual_mode=residual_mode,  # type: ignore[arg-type]
                               activation=ReLU)

    decoder_output = decoder(randn(batch_size, in_channels, *input_shape))
    skip = randn(batch_size, out_channels,
                 input_shape[0] * stride[0], input_shape[1] * stride[1])
    assert skip.shape == decoder_output.shape
    assert merge_block(decoder_output, skip).shape == skip.shape


def test_all_merge_modes_produce_same_output_shape() -> None:
    """All four modes must produce identical output shapes for the same inputs."""
    x = randn(2, 8, 16, 16)
    skip = randn(2, 8, 16, 16)
    params = get_encoder_params_2d(kernel=(3, 3), stride=(1, 1), dilation=(1, 1))
    shapes = set()
    for mode in _ALL_MODES:
        block = Merge2DBlock(layer_number=0,
                             in_channels_from_skip=8,
                             in_channels_from_decoder=8,
                             out_channels=8,
                             conv_params=params,
                             residual_mode=mode,  # type: ignore[arg-type]
                             activation=ReLU)
        shapes.add(block(x, skip).shape)
    assert len(shapes) == 1, f"Modes produced different shapes: {shapes}"


# ---------------------------------------------------------------------------
# Asymmetric input channels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("residual_mode", _ALL_MODES)
def test_merge2d_block_asymmetric_in_channels(residual_mode: str) -> None:
    """Block works when skip and decoder have different channel counts."""
    block = _make_merge(residual_mode, in_channels_from_skip=8, in_channels_from_decoder=16, out_channels=4)
    out = block(randn(1, 16, 16, 16), randn(1, 8, 16, 16))
    assert out.shape == (1, 4, 16, 16)


# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("residual_mode", _ALL_MODES)
def test_merge2d_block_backward_pass(residual_mode: str) -> None:
    """Backward pass produces finite gradients for all modes."""
    block = _make_merge(residual_mode, in_channels_from_skip=8, in_channels_from_decoder=8, out_channels=8,
                        kernel=(3, 3), stride=(1, 1))
    x = randn(2, 8, 16, 16, requires_grad=True)
    skip = randn(2, 8, 16, 16)
    loss = sqrt(MSELoss()(block(x, skip), randn(2, 8, 16, 16)) + 1e-8)
    loss.backward()
    assert isfinite(loss).item()
    assert x.grad is not None
    assert isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# Metadata / flag tests
# ---------------------------------------------------------------------------

def test_merge2d_block_merging_layer_flag() -> None:
    """merging_layer must be True so the UNet forward pass can detect merge blocks."""
    block = _make_merge("he2")
    assert block.merging_layer is True


def test_merge2d_block_invalid_residual_mode_raises_at_construction() -> None:
    """Unknown residual_mode raises ValueError at construction (registry lookup)."""
    params = get_encoder_params_2d(kernel=(3, 3), stride=(1, 1), dilation=(1, 1))
    with pytest.raises(ValueError, match="invalid_mode"):
        Merge2DBlock(layer_number=0,
                     in_channels_from_skip=4,
                     in_channels_from_decoder=4,
                     out_channels=4,
                     conv_params=params,
                     residual_mode="invalid_mode")  # type: ignore[arg-type]


def test_merge2d_block_shape_mismatch_raises() -> None:
    """forward() must raise when decoder output and skip connection spatial dims differ."""
    block = _make_merge("he2")
    with pytest.raises(AssertionError):
        block(randn(1, 4, 16, 16), randn(1, 4, 32, 32))


# ---------------------------------------------------------------------------
# Structural tests — verify internal block wiring per mode
# ---------------------------------------------------------------------------

def test_merge2d_block_conv_x_conv_skip_no_bn_no_act() -> None:
    """conv_x and conv_skip are always bare linear convs (no BN, no activation) for all modes."""
    _mode_to_cls = {
        "local_refinement": LocalRefinementMerge,
        "he1": He1Merge,
        "he2": He2Merge,
        "attention_gate": AttentionGateMerge,
    }
    for mode, cls in _mode_to_cls.items():
        inner: LocalRefinementMerge | He1Merge | He2Merge | AttentionGateMerge = cls(  # type: ignore[assignment]
            **_inner_kwargs(mode))
        assert inner.conv_x.batchnorm2d is None, f"conv_x must not apply BN in mode '{mode}'"
        assert inner.conv_x.activation is None, f"conv_x must not apply activation in mode '{mode}'"
        assert inner.conv_skip.batchnorm2d is None, f"conv_skip must not apply BN in mode '{mode}'"
        assert inner.conv_skip.activation is None, f"conv_skip must not apply activation in mode '{mode}'"


def test_merge2d_block_local_refinement_structure() -> None:
    """local_refinement: BN+Act on merged sum; conv_refine has BN and activation (post-activation)."""
    inner = LocalRefinementMerge(**_inner_kwargs("local_refinement"))
    assert isinstance(inner.batchnorm2d_out, torch.nn.BatchNorm2d)
    assert isinstance(inner.activation, torch.nn.ReLU)
    assert inner.conv_refine.batchnorm2d is not None
    assert inner.conv_refine.activation is not None


def test_merge2d_block_he1_structure() -> None:
    """he1: pre-activation, one refinement conv that is a pure linear conv."""
    inner = He1Merge(**_inner_kwargs("he1"))
    assert isinstance(inner.batchnorm2d_out, torch.nn.BatchNorm2d)
    assert isinstance(inner.activation, torch.nn.ReLU)
    assert inner.conv_no_BNAct_refine.batchnorm2d is None
    assert inner.conv_no_BNAct_refine.activation is None


def test_merge2d_block_he2_structure() -> None:
    """he2: pre-activation, two refinement convs that are pure linear convs."""
    inner = He2Merge(**_inner_kwargs("he2"))
    assert isinstance(inner.batchnorm2d_out1, torch.nn.BatchNorm2d)
    assert isinstance(inner.activation1, torch.nn.ReLU)
    assert isinstance(inner.batchnorm2d_out2, torch.nn.BatchNorm2d)
    assert isinstance(inner.activation2, torch.nn.ReLU)
    assert inner.conv_no_BNAct_refine1.batchnorm2d is None
    assert inner.conv_no_BNAct_refine1.activation is None
    assert inner.conv_no_BNAct_refine2.batchnorm2d is None
    assert inner.conv_no_BNAct_refine2.activation is None


def test_merge2d_block_attention_gate_structure() -> None:
    """attention_gate: has an AttentionGate module and he2-style refinement convs."""
    from SkiNet.ML.model.blocks.merge2d_residual_blocks import AttentionGate
    inner = AttentionGateMerge(**_inner_kwargs("attention_gate"))
    assert isinstance(inner.attention_gate, AttentionGate)
    assert inner.conv_no_BNAct_refine1.batchnorm2d is None
    assert inner.conv_no_BNAct_refine1.activation is None
    assert inner.conv_no_BNAct_refine2.batchnorm2d is None
    assert inner.conv_no_BNAct_refine2.activation is None
