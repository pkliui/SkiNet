import pytest
import torch
from torch import isfinite, randn, sqrt
from torch.nn import MSELoss, ReLU

from SkiNet.ML.model.blocks.conv2d_layer import Conv2dLayer
from SkiNet.ML.model.blocks.encoder2d import Encoder2D
from SkiNet.ML.utils.sampling.encoder_sampling import get_encoder_params_2d

# ---------------------------------------------------------------------------
# Shared parametrize cases reused across mode-specific shape tests
# ---------------------------------------------------------------------------
_SHAPE_CASES = [
    # batch, in_ch, out_ch, in_size, expected_size, kernel, stride, dilation
    (1, 3, 6, 16, 16, 3, 1, 1),
    (6, 3, 6, 16, 16, 3, 1, 1),
    (20, 3, 6, 16, 16, 3, 1, 1),
    (1, 3, 6, 128, 128, 3, 1, 1),
    (1, 3, 6, 512, 512, 3, 1, 1),
    (1, 1, 2, 16, 16, 3, 1, 1),
    (1, 16, 32, 16, 16, 3, 1, 1),
    (1, 3, 3, 16, 16, 3, 1, 1),
    (1, 3, 3, 16, 16, 5, 1, 1),
    (1, 3, 3, 16, 16, 3, 1, 3),
    (1, 3, 3, 16, 16, 5, 1, 3),
    (1, 3, 6, 32, 32, 3, 1, 2),
    (1, 3, 6, 32, 32, 3, 1, 4),
    (1, 3, 6, 32, 32, 5, 1, 2),
    (1, 3, 6, 16, 8, 3, 2, 1),
    (6, 3, 6, 16, 8, 3, 2, 2),
    (20, 3, 6, 16, 8, 3, 2, 2),
    (1, 3, 6, 4, 2, 3, 2, 1),
    (1, 3, 6, 128, 64, 3, 2, 1),
    (1, 3, 6, 512, 256, 3, 2, 1),
    (1, 1, 6, 16, 8, 3, 2, 2),
    (1, 3, 6, 16, 8, 5, 2, 2),
    (1, 3, 6, 16, 8, 3, 2, 4),
    (1, 3, 6, 16, 8, 5, 2, 1),
    (1, 3, 6, 16, 8, 5, 2, 3),
]


def _make_encoder(in_channels: int, out_channels: int, kernel: int, stride: int, dilation: int,
                  use_residual: bool, mode: str, layer_number: int = 0) -> "Encoder2D":
    return Encoder2D(in_channels=in_channels,
                     out_channels=out_channels,
                     conv_params=get_encoder_params_2d(kernel=kernel, stride=stride, dilation=dilation),
                     use_residual=use_residual,
                     apply_bias=False,
                     activation=ReLU,
                     layer_number=layer_number,
                     residual_mode=mode)  # type: ignore[arg-type]


class TestEncoder2D:

    # ------------------------------------------------------------------
    # Forward pass shape tests — parametrised over all three residual modes
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("batch_size, in_channels, out_channels, input_size, expected_size, kernel, stride, dilation",
                             _SHAPE_CASES)
    def test_encoder2d_he2_forward_pass_shapes(self, batch_size: int, in_channels: int, out_channels: int,
                                               input_size: int, expected_size: int, kernel: int, stride: int, dilation: int) -> None:
        """Output shape is correct for he2 mode across all parameter combinations."""
        x = randn(batch_size, in_channels, input_size, input_size)
        enc = _make_encoder(in_channels, out_channels, kernel, stride, dilation, True, "he2")
        assert enc(x).shape == (batch_size, out_channels, expected_size, expected_size)

    @pytest.mark.parametrize("batch_size, in_channels, out_channels, input_size, expected_size, kernel, stride, dilation",
                             _SHAPE_CASES)
    def test_encoder2d_local_refinement_forward_pass_shapes(self, batch_size: int, in_channels: int, out_channels: int,
                                                            input_size: int, expected_size: int, kernel: int, stride: int, dilation: int) -> None:
        """Output shape is correct for local_refinement mode across all parameter combinations."""
        x = randn(batch_size, in_channels, input_size, input_size)
        enc = _make_encoder(in_channels, out_channels, kernel, stride, dilation, True, "local_refinement")
        assert enc(x).shape == (batch_size, out_channels, expected_size, expected_size)

    @pytest.mark.parametrize("batch_size, in_channels, out_channels, input_size, expected_size, kernel, stride, dilation",
                             _SHAPE_CASES)
    def test_encoder2d_se_forward_pass_shapes(self, batch_size: int, in_channels: int, out_channels: int,
                                              input_size: int, expected_size: int, kernel: int, stride: int, dilation: int) -> None:
        """Output shape is correct for se mode across all parameter combinations."""
        x = randn(batch_size, in_channels, input_size, input_size)
        enc = _make_encoder(in_channels, out_channels, kernel, stride, dilation, True, "se")
        assert enc(x).shape == (batch_size, out_channels, expected_size, expected_size)

    # ------------------------------------------------------------------
    # Backward pass tests
    # ------------------------------------------------------------------

    def test_encoder2d_local_refinement_backward_pass(self) -> None:
        """local_refinement backward: loss finite, output non-negative (post-activation)."""
        x = randn(2, 3, 16, 16)
        enc = _make_encoder(3, 6, 3, 1, 1, True, "local_refinement")
        output = enc(x)
        loss = sqrt(MSELoss()(output, randn(2, 6, 16, 16)) + 1e-8)
        loss.backward()

        assert output.shape == (2, 6, 16, 16)
        assert isfinite(loss).item()
        assert (output >= 0.0).all()
        assert isfinite(output).all()

    def test_encoder2d_he2_backward_pass(self) -> None:
        """he2 backward: loss finite, output finite (no final ReLU so sign unconstrained)."""
        x = randn(2, 3, 16, 16)
        enc = _make_encoder(3, 6, 3, 1, 1, True, "he2")
        output = enc(x)
        loss = sqrt(MSELoss()(output, randn(2, 6, 16, 16)) + 1e-8)
        loss.backward()

        assert output.shape == (2, 6, 16, 16)
        assert isfinite(loss).item()
        assert isfinite(output).all()

    def test_encoder2d_se_backward_pass(self) -> None:
        """se backward: loss finite, output finite (pre-activation, sign unconstrained)."""
        x = randn(2, 3, 16, 16)
        enc = _make_encoder(3, 16, 3, 1, 1, True, "se")
        output = enc(x)
        loss = sqrt(MSELoss()(output, randn(2, 16, 16, 16)) + 1e-8)
        loss.backward()

        assert output.shape == (2, 16, 16, 16)
        assert isfinite(loss).item()
        assert isfinite(output).all()

    # ------------------------------------------------------------------
    # Computation graph: local_refinement vs he2 are structurally distinct
    # ------------------------------------------------------------------

    def test_local_refinement_output_is_non_negative(self) -> None:
        """local_refinement ends with an activation inside conv_refine → output always >= 0."""
        enc = _make_encoder(3, 16, 3, 2, 1, True, "local_refinement")
        with torch.no_grad():
            out = enc(randn(4, 3, 32, 32))
        assert (out >= 0.0).all(), "local_refinement must produce non-negative outputs (post-activation)"

    def test_he2_output_can_be_negative(self) -> None:
        """he2 has no activation after the final addition → output sign is unconstrained."""
        torch.manual_seed(0)
        enc = _make_encoder(3, 16, 3, 2, 1, True, "he2")
        with torch.no_grad():
            out = enc(randn(8, 3, 32, 32))
        assert (out < 0.0).any(), "he2 should produce some negative outputs (pre-activation, no final ReLU)"

    def test_se_output_can_be_negative(self) -> None:
        """se (pre-activation) has no activation after shortcut addition → sign unconstrained."""
        torch.manual_seed(0)
        enc = _make_encoder(3, 16, 3, 2, 1, True, "se")
        with torch.no_grad():
            out = enc(randn(8, 3, 32, 32))
        assert (out < 0.0).any(), "se should produce some negative outputs (pre-activation)"

    # ------------------------------------------------------------------
    # he2-specific structural tests
    # ------------------------------------------------------------------

    def test_encoder2d_he2_requires_use_residual(self) -> None:
        """he2 requires use_residual=True and raises ValueError at construction otherwise."""
        with pytest.raises(ValueError, match="use_residual=False"):
            _make_encoder(3, 6, 3, 2, 1, False, "he2")

    @pytest.mark.parametrize("in_channels, out_channels, stride", [
        (3, 6, 1),
        (3, 6, 2),
        (8, 8, 2),
        (8, 16, 2),
    ])
    def test_encoder2d_he2_shortcut_is_always_projection(self, in_channels: int, out_channels: int, stride: int) -> None:
        """he2 shortcut is always a 1×1 Conv2dLayer projection."""
        enc = _make_encoder(in_channels, out_channels, 3, stride, 1, True, "he2")
        shortcut = enc._block.shortcut
        assert isinstance(shortcut, Conv2dLayer)
        assert shortcut.conv2d.kernel_size == (1, 1)

    def test_encoder2d_he2_residual_gradient_flows_through_shortcut(self) -> None:
        """Projection shortcut weights receive non-zero gradient after backward."""
        enc = _make_encoder(3, 8, 3, 2, 1, True, "he2")
        MSELoss()(enc(randn(2, 3, 16, 16)), randn(2, 8, 8, 8)).backward()
        grad = enc._block.shortcut.conv2d.weight.grad  # type: ignore[union-attr]
        assert grad is not None
        assert grad.abs().sum().item() > 0.0  # type: ignore[operator]

    # ------------------------------------------------------------------
    # se-specific structural tests
    # ------------------------------------------------------------------

    def test_encoder2d_se_requires_use_residual(self) -> None:
        """se requires use_residual=True and raises ValueError at construction otherwise."""
        with pytest.raises(ValueError, match="use_residual=False"):
            _make_encoder(3, 6, 3, 2, 1, False, "se")

    def test_encoder2d_se_shortcut_is_projection(self) -> None:
        """se shortcut is a 1×1 Conv2dLayer projection (same as he2)."""
        enc = _make_encoder(3, 16, 3, 2, 1, True, "se")
        shortcut = enc._block.shortcut
        assert isinstance(shortcut, Conv2dLayer)
        assert shortcut.conv2d.kernel_size == (1, 1)

    def test_encoder2d_se_channel_attention_gradient_flows(self) -> None:
        """ChannelAttention FC weights receive non-zero gradient after backward."""
        enc = _make_encoder(3, 16, 3, 2, 1, True, "se")
        MSELoss()(enc(randn(2, 3, 16, 16)), randn(2, 16, 8, 8)).backward()
        for p in enc._block.channel_attention.parameters():  # type: ignore[union-attr]
            assert p.grad is not None
            assert p.grad.abs().sum().item() > 0.0

    # ------------------------------------------------------------------
    # local_refinement-specific structural tests
    # ------------------------------------------------------------------

    def test_encoder2d_local_refinement_use_residual_false_same_shape(self) -> None:
        """local_refinement with use_residual=False produces the same shape as use_residual=True."""
        enc_with = _make_encoder(3, 6, 3, 2, 1, True, "local_refinement")
        enc_without = _make_encoder(3, 6, 3, 2, 1, False, "local_refinement")
        x = randn(1, 3, 16, 16)
        assert enc_with(x).shape == enc_without(x).shape

    # ------------------------------------------------------------------
    # Construction-time validation
    # ------------------------------------------------------------------

    def test_encoder2d_invalid_residual_mode_raises_at_construction(self) -> None:
        """Unknown residual_mode raises ValueError at construction (registry lookup)."""
        with pytest.raises(ValueError, match="unknown_mode"):
            _make_encoder(3, 6, 3, 1, 1, True, "unknown_mode")

    def test_encoder2d_merging_layer_flag_is_false(self) -> None:
        """merging_layer must be False so the UNet forward pass does not treat encoders as merge blocks."""
        for mode in ("local_refinement", "he2", "se"):
            enc = _make_encoder(3, 16, 3, 2, 1, True, mode)
            assert enc.merging_layer is False
