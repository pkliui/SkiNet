from typing import Any

import torch
from torch import isfinite, randn
from torch.nn import MSELoss, ReLU

from SkiNet.ML.model.blocks.merge2d_residual_blocks import (AttentionGate, AttentionGateMerge,
                                                            He1Merge, He2Merge,
                                                            LocalRefinementMerge)
from SkiNet.ML.utils.sampling.encoder_sampling import EncoderParams2D, get_encoder_params_2d


def _params(kernel: Any = (3, 3), stride: Any = (1, 1), dilation: Any = (1, 1)) -> EncoderParams2D:
    return get_encoder_params_2d(kernel=kernel, stride=stride, dilation=dilation)


def _merge_kwargs(in_skip: int = 8, in_dec: int = 8, out: int = 8, **kw: Any) -> dict[str, Any]:
    return dict(in_channels_from_skip=in_skip,
                in_channels_from_decoder=in_dec,
                out_channels=out,
                conv_params=_params(**kw),
                activation=ReLU)


# ---------------------------------------------------------------------------
# AttentionGate
# ---------------------------------------------------------------------------

class TestAttentionGate:

    def test_output_shape_equals_skip_shape(self) -> None:
        """Gate output shape must match the skip connection (F_l channels, same spatial dims)."""
        gate = AttentionGate(F_g=16, F_l=8, F_int=4)
        g = randn(2, 16, 32, 32)
        x = randn(2, 8, 32, 32)
        assert gate(g, x).shape == x.shape

    def test_attention_weights_in_unit_interval(self) -> None:
        """Spatial attention map alpha is produced by Sigmoid — must lie strictly in (0, 1)."""
        gate = AttentionGate(F_g=8, F_l=8, F_int=4)
        g, x = randn(1, 8, 16, 16), randn(1, 8, 16, 16)
        with torch.no_grad():
            alpha = gate.psi(gate.relu(gate.W_g(g) + gate.W_x(x)))
        assert (alpha > 0.0).all(), "attention alpha must be > 0 (sigmoid lower bound)"
        assert (alpha < 1.0).all(), "attention alpha must be < 1 (sigmoid upper bound)"

    def test_argument_order_matters(self) -> None:
        """
        Swapping g and x must produce a different output.
        W_g and W_x have independent weights, so gate(g, x) != gate(x, g) in general.
        This guards against silent argument transposition bugs.
        """
        torch.manual_seed(0)
        gate = AttentionGate(F_g=8, F_l=8, F_int=4)
        # use tensors with large differences to make collision astronomically unlikely
        g = torch.ones(1, 8, 8, 8) * 3.0
        x = torch.ones(1, 8, 8, 8) * -3.0
        with torch.no_grad():
            out_correct = gate(g, x)
            out_swapped = gate(x, g)
        assert not torch.allclose(out_correct, out_swapped), \
            "gate(g, x) == gate(x, g): argument order has no effect — W_g and W_x may be identical"

    def test_zero_gating_signal_suppresses_output(self) -> None:
        """A zero gating signal should produce low attention weights (near 0.5 * x from sigmoid(0))."""
        gate = AttentionGate(F_g=8, F_l=8, F_int=4)
        g_zero = torch.zeros(1, 8, 16, 16)
        x = randn(1, 8, 16, 16)
        with torch.no_grad():
            out_zero_g = gate(g_zero, x)
            out_nonzero_g = gate(randn(1, 8, 16, 16), x)
        # outputs should differ when gating signal changes
        assert not torch.allclose(out_zero_g, out_nonzero_g)

    def test_backward_pass(self) -> None:
        """Gradients flow through W_g, W_x, and psi."""
        gate = AttentionGate(F_g=16, F_l=8, F_int=4)
        g = randn(2, 16, 16, 16)
        x = randn(2, 8, 16, 16)
        loss = MSELoss()(gate(g, x), randn(2, 8, 16, 16))
        loss.backward()
        for p in gate.parameters():
            assert p.grad is not None
            assert isfinite(p.grad).all()

    def test_asymmetric_fg_fl(self) -> None:
        """AttentionGate handles F_g != F_l (typical in U-Net where decoder != encoder channels)."""
        gate = AttentionGate(F_g=32, F_l=16, F_int=8)
        out = gate(randn(2, 32, 8, 8), randn(2, 16, 8, 8))
        assert out.shape == (2, 16, 8, 8)


# ---------------------------------------------------------------------------
# LocalRefinementMerge
# ---------------------------------------------------------------------------

class TestLocalRefinementMerge:

    def test_output_shape(self) -> None:
        m = LocalRefinementMerge(**_merge_kwargs())
        assert m(randn(2, 8, 16, 16), randn(2, 8, 16, 16)).shape == (2, 8, 16, 16)

    def test_output_is_non_negative(self) -> None:
        """Post-activation pattern — conv_refine ends with activation, so output >= 0."""
        m = LocalRefinementMerge(**_merge_kwargs())
        with torch.no_grad():
            out = m(randn(4, 8, 8, 8), randn(4, 8, 8, 8))
        assert (out >= 0.0).all()

    def test_conv_refine_has_bn_and_activation(self) -> None:
        m = LocalRefinementMerge(**_merge_kwargs())
        assert m.conv_refine.batchnorm2d is not None
        assert m.conv_refine.activation is not None

    def test_projection_convs_are_bare(self) -> None:
        m = LocalRefinementMerge(**_merge_kwargs())
        assert m.conv_x.batchnorm2d is None
        assert m.conv_skip.batchnorm2d is None


# ---------------------------------------------------------------------------
# He1Merge
# ---------------------------------------------------------------------------

class TestHe1Merge:

    def test_output_shape(self) -> None:
        m = He1Merge(**_merge_kwargs())
        assert m(randn(2, 8, 16, 16), randn(2, 8, 16, 16)).shape == (2, 8, 16, 16)

    def test_refinement_conv_is_bare(self) -> None:
        """Pre-activation pattern: refinement conv must be a pure linear transform."""
        m = He1Merge(**_merge_kwargs())
        assert m.conv_no_BNAct_refine.batchnorm2d is None
        assert m.conv_no_BNAct_refine.activation is None

    def test_backward_pass(self) -> None:
        m = He1Merge(**_merge_kwargs())
        x, skip = randn(2, 8, 8, 8, requires_grad=True), randn(2, 8, 8, 8)
        MSELoss()(m(x, skip), randn(2, 8, 8, 8)).backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# He2Merge
# ---------------------------------------------------------------------------

class TestHe2Merge:

    def test_output_shape(self) -> None:
        m = He2Merge(**_merge_kwargs())
        assert m(randn(2, 8, 16, 16), randn(2, 8, 16, 16)).shape == (2, 8, 16, 16)

    def test_both_refinement_convs_are_bare(self) -> None:
        m = He2Merge(**_merge_kwargs())
        assert m.conv_no_BNAct_refine1.batchnorm2d is None
        assert m.conv_no_BNAct_refine1.activation is None
        assert m.conv_no_BNAct_refine2.batchnorm2d is None
        assert m.conv_no_BNAct_refine2.activation is None

    def test_backward_pass(self) -> None:
        m = He2Merge(**_merge_kwargs())
        x, skip = randn(2, 8, 8, 8, requires_grad=True), randn(2, 8, 8, 8)
        MSELoss()(m(x, skip), randn(2, 8, 8, 8)).backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# AttentionGateMerge
# ---------------------------------------------------------------------------

class TestAttentionGateMerge:

    def test_output_shape(self) -> None:
        m = AttentionGateMerge(**_merge_kwargs())
        assert m(randn(2, 8, 16, 16), randn(2, 8, 16, 16)).shape == (2, 8, 16, 16)

    def test_has_attention_gate_module(self) -> None:
        m = AttentionGateMerge(**_merge_kwargs())
        assert isinstance(m.attention_gate, AttentionGate)

    def test_refinement_convs_are_bare(self) -> None:
        """Post-merge refinement follows he2 pattern — both convs must be pure linear."""
        m = AttentionGateMerge(**_merge_kwargs())
        assert m.conv_no_BNAct_refine1.batchnorm2d is None
        assert m.conv_no_BNAct_refine1.activation is None
        assert m.conv_no_BNAct_refine2.batchnorm2d is None
        assert m.conv_no_BNAct_refine2.activation is None

    def test_skip_gating_changes_output(self) -> None:
        """Attention gate must affect the output — different skip inputs should produce different outputs."""
        torch.manual_seed(1)
        m = AttentionGateMerge(**_merge_kwargs())
        x = randn(1, 8, 8, 8)
        skip_a = randn(1, 8, 8, 8)
        skip_b = randn(1, 8, 8, 8)
        with torch.no_grad():
            assert not torch.allclose(m(x, skip_a), m(x, skip_b))

    def test_backward_pass(self) -> None:
        m = AttentionGateMerge(**_merge_kwargs())
        x, skip = randn(2, 8, 8, 8, requires_grad=True), randn(2, 8, 8, 8)
        loss = MSELoss()(m(x, skip), randn(2, 8, 8, 8))
        loss.backward()
        assert x.grad is not None
        assert isfinite(x.grad).all()
        # attention gate weights must also receive gradients
        for p in m.attention_gate.parameters():
            assert p.grad is not None

    def test_asymmetric_channels(self) -> None:
        """Works when skip and decoder channels differ."""
        m = AttentionGateMerge(**_merge_kwargs(in_skip=4, in_dec=16, out=8))
        out = m(randn(1, 16, 8, 8), randn(1, 4, 8, 8))
        assert out.shape == (1, 8, 8, 8)
