from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from SkiNet.ML.model.architecture.unet2d import DecoderPath, UNet2D

torch.manual_seed(0)


# --------------------------------------------------
# Mock objects
# --------------------------------------------------

MOCK_DEC_OUT_CHANNELS = 4
class MockDecoder(nn.Module):
    """
    Mock decoder for structural testing of DecoderPath.
    """

    def __init__(self, layer_num: int, out_channels: int = MOCK_DEC_OUT_CHANNELS) -> None:
        super().__init__()
        self._layer_num = layer_num
        self._out_channels = out_channels

    @property
    def layer_number(self) -> int:
        return self._layer_num

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x  # No-op


class MockMergeBlock(nn.Module):
    """
    Mock merge block for structural testing of DecoderPath.
    """

    def __init__(self, layer_num: int) -> None:
        super().__init__()
        self._layer_num = layer_num

    @property
    def layer_number(self) -> int:
        return self._layer_num

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return x  # No-op


# --------------------------------------------------
# Test DecoderPath constructor validation logic
# --------------------------------------------------

def test_decoder_path_validates_count_mismatch() -> None:
    """
    Invariant 1: Pairing count. DecoderPath should reject mismatched decoder/merge counts.
    """
    decoders = nn.ModuleList([MockDecoder(layer_num=4)])
    mergeblocks = nn.ModuleList([MockMergeBlock(layer_num=4), MockMergeBlock(layer_num=3)])  # Extra merge block

    with pytest.raises(ValueError, match="Decoders and merge blocks must be paired 1:1"):
        DecoderPath(decoders=decoders, mergeblocks=mergeblocks, out_channels=MOCK_DEC_OUT_CHANNELS)

def test_decoder_path_validates_layer_number_values() -> None:
    """
    Invariant 2: Layer number alignment. DecoderPath should reject decoder/merge layer numbers < 2.
    """
    decoders = nn.ModuleList([MockDecoder(layer_num=1), MockDecoder(layer_num=2), MockDecoder(layer_num=3)])  # layer 1
    mergeblocks = nn.ModuleList([MockMergeBlock(layer_num=2), MockMergeBlock(layer_num=3), MockMergeBlock(layer_num=4)])

    with pytest.raises(ValueError, match="Decoders and merge blocks must be at layers 2..N."):
        DecoderPath(decoders=decoders, mergeblocks=mergeblocks, out_channels=MOCK_DEC_OUT_CHANNELS)

def test_decoder_path_validates_layer_number_alignment() -> None:
    """
    Invariant 2: Layer number alignment. DecoderPath should reject misaligned decoder/merge layer numbers.
    """
    decoders = nn.ModuleList([MockDecoder(layer_num=2), MockDecoder(layer_num=3), MockDecoder(layer_num=4)])
    mergeblocks = nn.ModuleList([MockMergeBlock(layer_num=2), MockMergeBlock(layer_num=3), MockMergeBlock(layer_num=5)])  # skipped layer 4

    with pytest.raises(ValueError, match="Layer number mismatch at index"):
        DecoderPath(decoders=decoders, mergeblocks=mergeblocks, out_channels=MOCK_DEC_OUT_CHANNELS)

def test_decoder_path_accepts_valid_configuration() -> None:
    """
    DecoderPath should accept valid decoder/merge configurations.
    """
    decoders = nn.ModuleList([MockDecoder(layer_num=4, out_channels=8),
                              MockDecoder(layer_num=3, out_channels=4),
                              MockDecoder(layer_num=2, out_channels=2)])
    mergeblocks = nn.ModuleList([MockMergeBlock(layer_num=4),
                                 MockMergeBlock(layer_num=3),
                                 MockMergeBlock(layer_num=2)])

    dp = DecoderPath(decoders=decoders, mergeblocks=mergeblocks, out_channels=2)

    assert len(dp.decoders) == 3
    assert len(dp.mergeblocks) == 3
    assert dp.out_channels == 2

def test_decoder_path_validates_output_channel_consistency() -> None:
    """
    Invariant 3: Output channel consistency (if decoders is non-empty). DecoderPath should reject output channel mismatches.
    """
    decoders = nn.ModuleList([MockDecoder(layer_num=2, out_channels=16)])
    mergeblocks = nn.ModuleList([MockMergeBlock(layer_num=2)])

    # out_channels=32 but shallowest decoder has out_channels=16
    with pytest.raises(ValueError, match="Output channel mismatch"):
        DecoderPath(decoders=decoders, mergeblocks=mergeblocks, out_channels=32)

# --------------------------------------------------
# Check UNet layers length
# --------------------------------------------------

def test_unet_layers_n4_forward() -> None:
    """
    Test UNet2D layer counts for a specific configuration.
    """
    model = UNet2D(in_channels=3, out_channels_layer1=8, number_of_layers=4, num_output_classes=1)
    assert len(model.encoders) == 4
    assert len(model.decoders) == 3
    assert len(model.mergeblocks) == 3

# --------------------------------------------------
# UNet forward shape tests
# --------------------------------------------------
@pytest.mark.unet2d
@pytest.mark.parametrize(
    "batch_size,in_channels,out_channels_layer1,num_layers,num_classes,input_size",
    [
        (1, 1, 4, 2, 1, 64),  # shallow network, num_layers=2
        (1, 1, 4, 4, 1, 64),  # vary num_layers
        (1, 1, 4, 5, 1, 64),  #
        (1, 3, 4, 5, 1, 64),  # vary input channels
        (1, 3, 8, 5, 1, 64),  # vary y channels
        (4, 3, 4, 5, 1, 64),  # vary batch
        (8, 3, 4, 5, 1, 64),  #
        (4, 3, 4, 5, 1, 32),  # vary input_size
        pytest.param(4, 3, 4, 5, 1, 512, marks=pytest.mark.slow),
        pytest.param(4, 3, 4, 5, 1, 1024, marks=pytest.mark.slow),
        (4, 3, 4, 5, 2, 256),  # vary number of classes
        (1, 1, 2, 2, 1, 16),  # minimum depth, small input
    ],
)
def test_unet2d_forward_shape(batch_size: int,
                              in_channels: int,
                              out_channels_layer1: int,
                              num_layers: int,
                              num_classes: int,
                              input_size: int) -> None:
    """
    Test the forward pass shape of the UNet2D model.
    """
    x = torch.randn(batch_size, in_channels, input_size, input_size)

    model = UNet2D(in_channels=in_channels,
                   out_channels_layer1=out_channels_layer1,
                   number_of_layers=num_layers,
                   num_output_classes=num_classes)
    # inference mode: makes BatchNorm/Dropout deterministic and avoids updating running stats
    model.eval()

    # disable grad for this test since we're only checking y shape, not gradients
    with torch.no_grad():
        y = model(x)

    assert y.shape == (batch_size, num_classes, input_size, input_size)

# --------------------------------------------------
# UNet gradient, backward test
# --------------------------------------------------
@pytest.mark.unet2d
@pytest.mark.parametrize(
    "batch_size,in_channels,out_channels_layer1,num_layers,num_classes,input_size",
    [
        (1, 1, 4, 2, 1, 64),  # shallow network, num_layers=2
        (1, 1, 4, 4, 1, 64),  # vary num_layers
        (1, 1, 4, 5, 1, 64),  #
        (1, 3, 4, 5, 1, 64),  # vary input channels
        (1, 3, 8, 5, 1, 64),  # vary y channels
        (4, 3, 4, 5, 1, 64),  # vary batch
        (8, 3, 4, 5, 1, 64),  #
        (4, 3, 4, 5, 1, 32),  # vary input_size
        pytest.param(4, 3, 4, 5, 1, 512, marks=pytest.mark.slow),
        (4, 3, 4, 5, 2, 256),  # vary number of classes
        (1, 1, 2, 2, 1, 16),  # minimum depth, small input
    ],
)
def test_unet2d_backward_pass(batch_size: int,
                              in_channels: int,
                              out_channels_layer1: int,
                              num_layers: int,
                              num_classes: int,
                              input_size: int) -> None:
    """
    Test the backward pass of the UNet2D model.
    """
    x = torch.randn(batch_size, in_channels, input_size, input_size).float()
    ground_truth = torch.randn(batch_size, num_classes, input_size, input_size).float()

    model = UNet2D(in_channels=in_channels,
                   out_channels_layer1=out_channels_layer1,
                   number_of_layers=num_layers,
                   num_output_classes=num_classes)
    model.train()  # explicitly enable training mode to allow gradients to be computed
    y = model(x)

    # Check output torch.Tensor size
    assert y.shape == (batch_size, num_classes, input_size, input_size)

    loss = torch.sqrt(nn.MSELoss()(y, ground_truth) + 1e-8)
    loss.backward()

    # Loss is a positive finite value
    assert loss.item() > 0
    assert torch.isfinite(loss).item() is True

    # Ensure all non-None gradients of parameters that require grad are finite
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads), "No gradients were computed for any parameter"
    assert all(torch.isfinite(g).all() for g in grads if g is not None)


# --------------------------------------------------
# Numerical stability
# --------------------------------------------------

@pytest.mark.parametrize("input_value", [1e6, 1e-6])
def test_unet2d_numerical_stability(input_value: float) -> None:
    """
    Test the numerical stability of the UNet2D model with extreme input values.
    """
    batch_size = 1
    in_channels = 2
    num_classes = 1

    x = torch.full((batch_size, in_channels, 64, 64), input_value)
    ground_truth = torch.full((batch_size, num_classes, 64, 64), input_value)

    model = UNet2D(in_channels=in_channels, out_channels_layer1=2, number_of_layers=5, num_output_classes=num_classes)
    model.train()
    y = model(x)

    # Check that output is finite (not NaN or Inf) even with extreme input values
    assert torch.isfinite(y).all()

    loss = torch.sqrt(nn.MSELoss()(y, ground_truth) + 1e-8)
    loss.backward()

    # Loss is a positive finite value
    assert loss.item() > 0
    assert torch.isfinite(loss).item() is True

    # Ensure all non-None gradients of parameters that require grad are finite
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads), "No gradients were computed for any parameter"
    assert all(torch.isfinite(g).all() for g in grads if g is not None)


# --------------------------------------------------
# Skip connections
# --------------------------------------------------

@pytest.mark.parametrize("input_shape", [(1, 3, 64, 64), (1, 3, 128, 128)])
def test_unet2d_skip_connections_change_output(input_shape: tuple[int, int, int, int]) -> None:
    """
    Check skip connections actually have an effect on the output
    by comparing model outputs with and without skip features.
    """
    x = torch.randn(*input_shape)

    model = UNet2D(in_channels=3,
                   out_channels_layer1=8,
                   number_of_layers=4,
                   num_output_classes=2)
    model.eval()
    with torch.no_grad():
        output_with_skips = model(x)

    # assert that model has merge blocks to test on
    assert len(model.mergeblocks) > 0

    # Define a forward pre-hook to zero out skip features before they are merged
    # Accept (module, inputs) as required by register_forward_pre_hook
    def zero_skip_pre_hook(module: nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        # inputs is typically (decoder_features, skip_features, *extra)
        if len(inputs) >= 2 and isinstance(inputs[1], torch.Tensor):
            return (inputs[0], torch.zeros_like(inputs[1]), *inputs[2:])
        return inputs

    # Attach that pre-hook to every merge block in model.mergeblocks and return handles to remove them later
    handles = [merge.register_forward_pre_hook(zero_skip_pre_hook) for merge in model.mergeblocks]
    assert handles, "Expected at least one merge block hook"

    # Run the model again with the same input, but now skip features will be zeroed out before merging
    try:
        with torch.no_grad():
            output_no_skips = model(x)
    # Enforce removal of the hooks to avoid side effects on other tests
    finally:
        for h in handles:
            h.remove()
    assert output_with_skips.shape == output_no_skips.shape, "y shapes should match"

    # Compute absolute and relative delta between outputs with and without skips
    delta = (output_with_skips - output_no_skips).abs().mean()
    rel_delta = delta / output_with_skips.abs().mean().clamp_min(1e-6)
    # print("delta", delta.item(), "rel_delta", rel_delta.item())
    # Set the threshold at roughly 10% of min observed relative delta
    # Observed rel_delta is ~0.3; use a much smaller threshold to avoid flakiness
    assert rel_delta > 1e-2, f"Skip connections have negligible relative effect: rel_delta={rel_delta.item():.6g}"


# --------------------------------------------------
# Validation Method Unit Tests
# --------------------------------------------------

def test_validate_skip_keys_raises_on_missing_keys() -> None:
    """
    Test _validate_skip_keys in isolation raises ValueError when skip connections are missing.
    """
    model = UNet2D(in_channels=3, out_channels_layer1=8, number_of_layers=4, num_output_classes=1)
    model.eval()

    # Create invalid skip connections dict missing layer 2
    invalid_skips = {
        1: torch.randn(1, 8, 64, 64),
        # 2: missing!
        3: torch.randn(1, 16, 32, 32),
        # no layer 4 as expected
    }

    with pytest.raises(ValueError, match="Skip keys mismatch"):
        model._validate_skip_keys(invalid_skips)


def test_validate_skip_keys_raises_on_extra_keys() -> None:
    """
    Test _validate_skip_keys in isolation raises ValueError when skip connections have extra keys.
    """
    model = UNet2D(in_channels=3, out_channels_layer1=8, number_of_layers=4, num_output_classes=1)
    model.eval()

    # Create invalid skip connections dict with extra layer 5 (but number_of_layers=4)
    invalid_skips = {
        1: torch.randn(1, 8, 64, 64),
        2: torch.randn(1, 16, 32, 32),
        3: torch.randn(1, 32, 16, 16),
        5: torch.randn(1, 64, 8, 8),  # Extra key!
    }

    with pytest.raises(ValueError, match="Skip keys mismatch"):
        model._validate_skip_keys(invalid_skips)


def test_validate_skip_count_raises_on_mismatch() -> None:
    """
    Test _validate_skip_count in isolation raises ValueError when count doesn't match decoder count.
    """
    model = UNet2D(in_channels=3, out_channels_layer1=8, number_of_layers=4, num_output_classes=1)
    model.eval()

    # Only 2 skip connections but 3 decoders (number_of_layers-1)
    invalid_skips = {
        1: torch.randn(1, 8, 64, 64),
        2: torch.randn(1, 16, 32, 32),
    }

    with pytest.raises(ValueError, match="Skip count .* != decoder count"):
        model._validate_skip_count(invalid_skips)


def test_log_near_zero_skips_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    """
    Test _log_near_zero_skips in isolation logs warning message for near-zero skip connections.
    """
    import logging

    model = UNet2D(in_channels=3, out_channels_layer1=8, number_of_layers=4, num_output_classes=1)
    model.eval()

    # Create skip connections with one near-zero torch.Tensor
    skip_connections = {
        1: torch.randn(1, 8, 64, 64),
        2: torch.zeros(1, 16, 32, 32),  # Near-zero
        3: torch.randn(1, 32, 16, 16),
    }

    with caplog.at_level(logging.WARNING):
        model._log_near_zero_skips(skip_connections)

    # Check that warning was logged for layer 2
    assert any((record.levelno == logging.WARNING) and ("Layer 2" in record.message) and ("near-zero magnitude" in record.message)
               for record in caplog.records), "Expected a WARNING about Layer 2 near-zero skip magnitude"

# --------------------------------------------------
# Residual mode combinations
# --------------------------------------------------

@pytest.mark.parametrize("encoder_residual_mode", ["he2", "local_refinement"])
@pytest.mark.parametrize("merge_residual_mode", ["he2", "he1", "local_refinement"])
def test_unet2d_residual_mode_combinations_forward(encoder_residual_mode: str,
                                                   merge_residual_mode: str) -> None:
    """All valid encoder × merge mode combinations produce the correct output shape."""
    x = torch.randn(1, 3, 64, 64)
    model = UNet2D(in_channels=3,
                   out_channels_layer1=4,
                   number_of_layers=4,
                   num_output_classes=1,
                   encoder_residual_mode=encoder_residual_mode,  # type: ignore[arg-type]
                   merge_residual_mode=merge_residual_mode)  # type: ignore[arg-type]
    model.eval()
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 64, 64)
    assert model.encoder_residual_mode == encoder_residual_mode
    assert model.merge_residual_mode == merge_residual_mode


# --------------------------------------------------
# Integration Tests: Validation During Forward Pass
# --------------------------------------------------

def test_forward_with_validation_disabled_does_not_validate() -> None:
    """
    With validate_forward=False, forward should not call any validation helpers.
    """
    x = torch.randn(1, 3, 64, 64)

    model = UNet2D(in_channels=3, out_channels_layer1=8, number_of_layers=4,
                   num_output_classes=1, validate_forward=False)
    model.eval()

    with patch.object(model, "_validate_skip_keys") as mock_keys, \
            patch.object(model, "_validate_skip_count") as mock_count, \
            patch.object(model, "_log_near_zero_skips") as mock_log, \
            torch.no_grad():
        y = model(x)

    assert y.shape == (1, 1, 64, 64)
    mock_keys.assert_not_called()
    mock_count.assert_not_called()
    mock_log.assert_not_called()


def test_forward_with_validation_enabled_calls_validators() -> None:
    """
    With validate_forward=True, forward should call all three validation helpers exactly once.
    """
    x = torch.randn(1, 3, 64, 64)

    model = UNet2D(in_channels=3, out_channels_layer1=8, number_of_layers=4,
                   num_output_classes=1, validate_forward=True)
    model.eval()

    with patch.object(model, "_validate_skip_keys") as mock_keys, \
            patch.object(model, "_validate_skip_count") as mock_count, \
            patch.object(model, "_log_near_zero_skips") as mock_log, \
            torch.no_grad():
        y = model(x)

    assert y.shape == (1, 1, 64, 64)
    mock_keys.assert_called_once()
    mock_count.assert_called_once()
    mock_log.assert_called_once()
