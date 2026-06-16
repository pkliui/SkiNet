import pytest
import torch
from torch import isfinite, randn
from torch.nn import MSELoss, ReLU
from torchvision.ops import SqueezeExcitation
from SkiNet.ML.model.blocks.encoder2d_residual_blocks import (ClassicalEncoder, He2Encoder, LocalRefinementEncoder,
                                                              SEEncoder)
from SkiNet.ML.utils.sampling.encoder_sampling import EncoderParams2D, get_encoder_params_2d


def _params(kernel: int = 3, stride: int = 1, dilation: int = 1) -> EncoderParams2D:
    return get_encoder_params_2d(kernel=kernel, stride=stride, dilation=dilation)

# ---------------------------------------------------------------------------
# ClassicalEncoder
# ---------------------------------------------------------------------------


class TestClassicalEncoder:

    def test_output_shape_no_downsample(self) -> None:
        enc = ClassicalEncoder(3, 16, _params(stride=1), False, ReLU, True)
        assert enc(randn(2, 3, 32, 32)).shape == (2, 16, 32, 32)

    def test_output_shape_with_downsample(self) -> None:
        enc = ClassicalEncoder(3, 16, _params(stride=2), False, ReLU, True)
        assert enc(randn(2, 3, 32, 32)).shape == (2, 16, 16, 16)

    def test_output_is_non_negative(self) -> None:
        """Post-activation pattern — both convs end with activation, so output >= 0."""
        enc = ClassicalEncoder(3, 16, _params(stride=2), False, ReLU, True)
        with torch.no_grad():
            out = enc(randn(4, 3, 32, 32))
        assert (out >= 0.0).all()

    def test_use_residual_ignored(self) -> None:
        """use_residual is accepted for registry compatibility but has no effect on output shape."""
        p = _params(stride=2)
        enc_true = ClassicalEncoder(3, 16, p, False, ReLU, True)
        enc_false = ClassicalEncoder(3, 16, p, False, ReLU, False)
        x = randn(1, 3, 32, 32)
        assert enc_true(x).shape == enc_false(x).shape

    def test_no_shortcut_attribute(self) -> None:
        """Classical encoder has no residual shortcut — no shortcut attribute must exist."""
        enc = ClassicalEncoder(3, 16, _params(stride=2), False, ReLU, True)
        assert not hasattr(enc, "shortcut")

    def test_both_convs_have_bn_and_activation(self) -> None:
        enc = ClassicalEncoder(3, 16, _params(stride=2), False, ReLU, True)
        assert enc.conv_downsample.batchnorm2d is not None
        assert enc.conv_downsample.activation is not None
        assert enc.conv_refine.batchnorm2d is not None
        assert enc.conv_refine.activation is not None

    def test_backward_pass(self) -> None:
        enc = ClassicalEncoder(3, 16, _params(stride=2), False, ReLU, True)
        loss = MSELoss()(enc(randn(2, 3, 16, 16)), randn(2, 16, 8, 8))
        loss.backward()
        assert all(p.grad is not None for p in enc.parameters())


# ---------------------------------------------------------------------------
# LocalRefinementEncoder
# ---------------------------------------------------------------------------


class TestLocalRefinementEncoder:

    def test_output_shape_no_downsample(self) -> None:
        enc = LocalRefinementEncoder(3, 16, _params(stride=1), False, ReLU, True)
        assert enc(randn(2, 3, 32, 32)).shape == (2, 16, 32, 32)

    def test_output_shape_with_downsample(self) -> None:
        enc = LocalRefinementEncoder(3, 16, _params(stride=2), False, ReLU, True)
        assert enc(randn(2, 3, 32, 32)).shape == (2, 16, 16, 16)

    def test_output_is_non_negative(self) -> None:
        """Post-activation pattern — output is always >= 0."""
        enc = LocalRefinementEncoder(3, 16, _params(stride=2), False, ReLU, True)
        with torch.no_grad():
            out = enc(randn(4, 3, 32, 32))
        assert (out >= 0.0).all()

    def test_use_residual_false_same_shape(self) -> None:
        """use_residual=False produces the same output shape as use_residual=True."""
        p = _params(stride=2)
        enc_r = LocalRefinementEncoder(3, 16, p, False, ReLU, True)
        enc_nr = LocalRefinementEncoder(3, 16, p, False, ReLU, False)
        x = randn(1, 3, 32, 32)
        assert enc_r(x).shape == enc_nr(x).shape

    def test_backward_pass(self) -> None:
        enc = LocalRefinementEncoder(3, 16, _params(stride=2), False, ReLU, True)
        loss = MSELoss()(enc(randn(2, 3, 16, 16)), randn(2, 16, 8, 8))
        loss.backward()
        assert all(p.grad is not None for p in enc.parameters())


# ---------------------------------------------------------------------------
# He2Encoder
# ---------------------------------------------------------------------------

class TestHe2Encoder:

    def test_output_shape_with_downsample(self) -> None:
        enc = He2Encoder(3, 16, _params(stride=2), False, ReLU, True)
        assert enc(randn(2, 3, 32, 32)).shape == (2, 16, 16, 16)

    def test_output_can_be_negative(self) -> None:
        """Pre-activation pattern — no final ReLU (it is applied before convolution), output sign unconstrained."""
        torch.manual_seed(0)
        enc = He2Encoder(3, 16, _params(stride=2), False, ReLU, True)
        with torch.no_grad():
            out = enc(randn(8, 3, 32, 32))
        assert (out < 0.0).any(), "he2 should produce some negative values (no final activation)"

    def test_requires_use_residual(self) -> None:
        with pytest.raises(ValueError, match="use_residual=False"):
            He2Encoder(3, 16, _params(stride=2), False, ReLU, False)

    def test_shortcut_is_1x1_projection(self) -> None:
        enc = He2Encoder(3, 16, _params(stride=2), False, ReLU, True)
        assert enc.shortcut.conv2d.kernel_size == (1, 1)

    def test_shortcut_gradient_flows(self) -> None:
        enc = He2Encoder(3, 16, _params(stride=2), False, ReLU, True)
        MSELoss()(enc(randn(2, 3, 16, 16)), randn(2, 16, 8, 8)).backward()
        grad = enc.shortcut.conv2d.weight.grad
        assert grad is not None
        assert grad.abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# SEEncoder
# ---------------------------------------------------------------------------

class TestSEEncoder:

    def test_output_shape_with_downsample(self) -> None:
        enc = SEEncoder(3, 16, _params(stride=2), False, ReLU, True)
        assert enc(randn(2, 3, 32, 32)).shape == (2, 16, 16, 16)

    def test_output_can_be_negative(self) -> None:
        """Pre-activation skeleton — output sign is unconstrained."""
        torch.manual_seed(0)
        enc = SEEncoder(3, 16, _params(stride=2), False, ReLU, True)
        with torch.no_grad():
            out = enc(randn(8, 3, 32, 32))
        assert (out < 0.0).any(), "se should produce some negative values (pre-activation)"

    def test_requires_use_residual(self) -> None:
        with pytest.raises(ValueError, match="use_residual=False"):
            SEEncoder(3, 16, _params(stride=2), False, ReLU, False)

    def test_has_channel_attention(self) -> None:
        enc = SEEncoder(3, 16, _params(stride=2), False, ReLU, True)
        assert isinstance(enc.channel_attention, SqueezeExcitation)

    def test_shortcut_is_1x1_projection(self) -> None:
        enc = SEEncoder(3, 16, _params(stride=2), False, ReLU, True)
        assert enc.shortcut.conv2d.kernel_size == (1, 1)

    def test_channel_attention_gradient_flows(self) -> None:
        enc = SEEncoder(3, 16, _params(stride=2), False, ReLU, True)
        MSELoss()(enc(randn(2, 3, 16, 16)), randn(2, 16, 8, 8)).backward()
        for p in enc.channel_attention.parameters():
            assert p.grad is not None
            assert p.grad.abs().sum().item() > 0.0

    @pytest.mark.parametrize("out_channels", [16, 32, 64, 128, 256])
    def test_se_valid_at_all_typical_encoder_widths(self, out_channels: int) -> None:
        """SE reduction floor ensures valid FC dims at every encoder layer width."""
        enc = SEEncoder(3, out_channels, _params(stride=2), False, ReLU, True)
        out = enc(randn(1, 3, 32, 32))
        assert out.shape == (1, out_channels, 16, 16)
        assert isfinite(out).all()

    def test_se_differs_from_he2_on_same_input(self) -> None:
        """SEEncoder and He2Encoder must produce different outputs — distinct computation graphs."""
        torch.manual_seed(7)
        se = SEEncoder(3, 16, _params(stride=1), False, ReLU, True)
        torch.manual_seed(7)
        he2 = He2Encoder(3, 16, _params(stride=1), False, ReLU, True)
        x = randn(2, 3, 16, 16)
        with torch.no_grad():
            assert not torch.allclose(se(x), he2(x)), \
                "SEEncoder and He2Encoder produced identical outputs — SE is not being applied"
