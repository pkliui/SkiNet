import pytest
from torch import nn, randn

from SkiNet.ML.model.blocks.decoder2d import Decoder2D
from SkiNet.ML.model.blocks.merge2d_block import Merge2DBlock
from SkiNet.ML.utils.sampling.decoder_sampling import get_decoder_params_2d
from SkiNet.ML.utils.sampling.encoder_sampling import get_encoder_params_2d
from SkiNet.ML.utils.typing_utils import TupleOfInt2d


@pytest.mark.parametrize("residual_mode", ["local_refinement", "he1", "he2"])
@pytest.mark.parametrize("batch_size, in_channels, out_channels, input_shape, kernel, stride, dilation", [
    (2, 8, 4, (32, 32), (3, 3), (2, 2), (1, 1)),
])
def test_merge2d_block_output_shape(batch_size: int,
                                    in_channels: int,
                                    out_channels: int,
                                    input_shape: TupleOfInt2d,
                                    kernel: TupleOfInt2d,
                                    stride: TupleOfInt2d,
                                    dilation: TupleOfInt2d,
                                    residual_mode: str) -> None:
    """Output shape matches (batch, out_channels, H*stride, W*stride) for all residual modes."""
    params = get_encoder_params_2d(kernel=kernel, stride=stride, dilation=dilation)

    decoder = Decoder2D(in_channels=in_channels,
                        out_channels=out_channels,
                        decoder_params=get_decoder_params_2d(params),
                        activation=nn.ReLU,
                        layer_number=1)

    merge_block = Merge2DBlock(layer_number=0,
                               in_channels_from_skip=out_channels,
                               in_channels_from_decoder=out_channels,
                               out_channels=out_channels,
                               conv_params=params,
                               residual_mode=residual_mode,  # type: ignore[arg-type]
                               activation=nn.ReLU)

    input_to_decoder = randn(batch_size, in_channels, input_shape[0], input_shape[1])
    skip_connection = randn(batch_size, out_channels, input_shape[0] * stride[0], input_shape[1] * stride[1])

    decoder_output = decoder(input_to_decoder)
    assert skip_connection.shape == decoder_output.shape

    output = merge_block(decoder_output, skip_connection)
    assert output.shape == (batch_size, out_channels, input_shape[0] * stride[0], input_shape[1] * stride[1])


@pytest.mark.parametrize("residual_mode", ["local_refinement", "he1", "he2"])
def test_merge2d_block_asymmetric_in_channels(residual_mode: str) -> None:
    """Block works when skip and decoder have different channel counts."""
    params = get_encoder_params_2d(kernel=(3, 3), stride=(2, 2), dilation=(1, 1))
    merge_block = Merge2DBlock(layer_number=0,
                               in_channels_from_skip=8,
                               in_channels_from_decoder=16,
                               out_channels=4,
                               conv_params=params,
                               residual_mode=residual_mode,  # type: ignore[arg-type]
                               activation=nn.ReLU)

    x = randn(1, 16, 16, 16)
    skip = randn(1, 8, 16, 16)
    output = merge_block(x, skip)
    assert output.shape == (1, 4, 16, 16)


def test_merge2d_block_merging_layer_flag() -> None:
    """merging_layer attribute must be True so the UNet forward pass can detect merge blocks."""
    params = get_encoder_params_2d(kernel=(3, 3), stride=(2, 2), dilation=(1, 1))
    merge_block = Merge2DBlock(layer_number=2,
                               in_channels_from_skip=4,
                               in_channels_from_decoder=4,
                               out_channels=4,
                               conv_params=params)
    assert merge_block.merging_layer is True


def test_merge2d_block_shape_mismatch_raises() -> None:
    """forward() must raise when decoder output and skip connection shapes differ."""
    params = get_encoder_params_2d(kernel=(3, 3), stride=(1, 1), dilation=(1, 1))
    merge_block = Merge2DBlock(layer_number=0,
                               in_channels_from_skip=4,
                               in_channels_from_decoder=4,
                               out_channels=4,
                               conv_params=params)

    x = randn(1, 4, 16, 16)
    skip = randn(1, 4, 32, 32)  # different spatial dims → should raise
    with pytest.raises(AssertionError):
        merge_block(x, skip)


def _make_merge_block(residual_mode: str) -> Merge2DBlock:
    params = get_encoder_params_2d(kernel=(3, 3), stride=(2, 2), dilation=(1, 1))
    return Merge2DBlock(layer_number=0,
                        in_channels_from_skip=4,
                        in_channels_from_decoder=4,
                        out_channels=4,
                        conv_params=params,
                        residual_mode=residual_mode,  # type: ignore[arg-type]
                        activation=nn.ReLU)


def test_merge2d_block_conv_x_conv_skip_no_bn_no_act() -> None:
    """conv_x and conv_skip are always bare linear convs (no BN, no activation) for all modes."""
    for mode in ("local_refinement", "he1", "he2"):
        block = _make_merge_block(mode)
        for attr in ("conv_x", "conv_skip"):
            conv = getattr(block, attr)
            assert conv.batchnorm2d is None, f"{attr} must not apply BN in mode '{mode}'"
            assert conv.activation is None, f"{attr} must not apply activation in mode '{mode}'"


def test_merge2d_block_local_refinement_structure() -> None:
    """
    local_refinement mode: BN+Act applied to merged sum, then conv_refine (with BN+Act) + residual.
    The refinement conv includes its own BN and activation (post-activation pattern).
    """
    block = _make_merge_block("local_refinement")
    assert isinstance(block.batchnorm2d_out, nn.BatchNorm2d)
    assert isinstance(block.activation, nn.ReLU)
    assert block.conv_refine.batchnorm2d is not None, "conv_refine must include BN"
    assert block.conv_refine.activation is not None, "conv_refine must include activation"


def test_merge2d_block_he1_structure() -> None:
    """
    he1 mode: pre-activation pattern with one refinement conv.
    conv_no_BNAct_refine must be a pure linear conv to preserve the identity shortcut.
    """
    block = _make_merge_block("he1")
    assert isinstance(block.batchnorm2d_out, nn.BatchNorm2d)
    assert isinstance(block.activation, nn.ReLU)
    assert block.conv_no_BNAct_refine.batchnorm2d is None, "refinement conv must not apply BN (pre-activation)"
    assert block.conv_no_BNAct_refine.activation is None, "refinement conv must not activate (identity shortcut)"


def test_merge2d_block_he2_structure() -> None:
    """
    he2 mode: pre-activation pattern with two refinement convs.
    Both refinement convs must be pure linear convs; BN+Act are applied externally.
    """
    block = _make_merge_block("he2")
    assert isinstance(block.batchnorm2d_out1, nn.BatchNorm2d)
    assert isinstance(block.activation1, nn.ReLU)
    assert isinstance(block.batchnorm2d_out2, nn.BatchNorm2d)
    assert isinstance(block.activation2, nn.ReLU)
    for attr in ("conv_no_BNAct_refine1", "conv_no_BNAct_refine2"):
        conv = getattr(block, attr)
        assert conv.batchnorm2d is None, f"{attr} must not apply BN"
        assert conv.activation is None, f"{attr} must not apply activation"
