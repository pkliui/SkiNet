import pytest
from torch import isfinite, randn, sqrt
from torch.nn import MSELoss, ReLU

from SkiNet.ML.model.blocks.conv2d_layer import Conv2dLayer
from SkiNet.ML.model.blocks.encoder2d import Encoder2D
from SkiNet.ML.utils.sampling.encoder_sampling import get_encoder_params_2d
from SkiNet.ML.utils.typing_utils import IntOrTuple2d


class TestEncoder2D:
    """
    Unit tests for Encoder2D class
    """

    # ------------------------------------------------------------------
    # Forward pass shape tests — parametrised over both residual modes
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("batch_size, in_channels, out_channels, input_size, expected_size, kernel, stride, dilation, use_residual",
                             [
                                 # SAME padding, stride=1
                                 (1, 3, 6, 16, 16, 3, 1, 1, True),
                                 (6, 3, 6, 16, 16, 3, 1, 1, True),
                                 (20, 3, 6, 16, 16, 3, 1, 1, True),
                                 (1, 3, 6, 128, 128, 3, 1, 1, True),
                                 (1, 3, 6, 512, 512, 3, 1, 1, True),
                                 # Vary channels
                                 (1, 1, 2, 16, 16, 3, 1, 1, True),
                                 (1, 16, 32, 16, 16, 3, 1, 1, True),
                                 # Vary kernel and dilation, stride=1
                                 (1, 3, 3, 16, 16, 3, 1, 1, True),
                                 (1, 3, 3, 16, 16, 5, 1, 1, True),
                                 (1, 3, 3, 16, 16, 3, 1, 3, True),
                                 (1, 3, 3, 16, 16, 5, 1, 3, True),
                                 (1, 3, 6, 32, 32, 3, 1, 2, True),
                                 (1, 3, 6, 32, 32, 3, 1, 4, True),
                                 (1, 3, 6, 32, 32, 5, 1, 2, True),
                                 # Downsampling, stride=2
                                 (1, 3, 6, 16, 8, 3, 2, 1, True),
                                 (6, 3, 6, 16, 8, 3, 2, 2, True),
                                 (20, 3, 6, 16, 8, 3, 2, 2, True),
                                 (1, 3, 6, 4, 2, 3, 2, 1, True),
                                 (1, 3, 6, 128, 64, 3, 2, 1, True),
                                 (1, 3, 6, 512, 256, 3, 2, 1, True),
                                 (1, 1, 6, 16, 8, 3, 2, 2, True),
                                 (1, 3, 6, 16, 8, 5, 2, 2, True),
                                 (1, 3, 6, 16, 8, 3, 2, 4, True),
                                 (1, 3, 6, 16, 8, 5, 2, 1, True),
                                 (1, 3, 6, 16, 8, 5, 2, 3, True),
                             ])
    def test_encoder2d_he2_forward_pass_shapes(self,
                                               batch_size: int,
                                               in_channels: int,
                                               out_channels: int,
                                               input_size: int,
                                               expected_size: int,
                                               kernel: IntOrTuple2d,
                                               stride: IntOrTuple2d,
                                               dilation: IntOrTuple2d,
                                               use_residual: bool) -> None:
        """Output shape is correct for he2 mode across all parameter combinations."""
        x = randn(batch_size, in_channels, input_size, input_size).float()
        encoder = Encoder2D(in_channels=in_channels,
                            out_channels=out_channels,
                            conv_params=get_encoder_params_2d(kernel=kernel, stride=stride, dilation=dilation),
                            use_residual=use_residual,
                            apply_bias=False,
                            activation=ReLU,
                            layer_number=0,
                            residual_mode="he2")
        assert encoder(x).shape == (batch_size, out_channels, expected_size, expected_size)

    @pytest.mark.parametrize("batch_size, in_channels, out_channels, input_size, expected_size, kernel, stride, dilation, use_residual",
                             [
                                 # SAME padding, stride=1, with and without residual
                                 (1, 3, 6, 16, 16, 3, 1, 1, True),
                                 (6, 3, 6, 16, 16, 3, 1, 1, True),
                                 (20, 3, 6, 16, 16, 3, 1, 1, True),
                                 (1, 3, 6, 128, 128, 3, 1, 1, True),
                                 (1, 3, 6, 512, 512, 3, 1, 1, True),
                                 (1, 1, 2, 16, 16, 3, 1, 1, True),
                                 (1, 16, 32, 16, 16, 3, 1, 1, True),
                                 (1, 3, 3, 16, 16, 3, 1, 1, True),
                                 (1, 3, 3, 16, 16, 5, 1, 1, True),
                                 (1, 3, 3, 16, 16, 3, 1, 3, True),
                                 (1, 3, 3, 16, 16, 5, 1, 3, True),
                                 (1, 3, 6, 32, 32, 3, 1, 2, True),
                                 (1, 3, 6, 32, 32, 3, 1, 4, True),
                                 (1, 3, 6, 32, 32, 5, 1, 2, True),
                                 # Without residual (valid for local_refinement)
                                 (1, 3, 3, 16, 16, 3, 1, 1, False),
                                 (1, 3, 3, 16, 16, 5, 1, 2, False),
                                 # Downsampling, stride=2
                                 (1, 3, 6, 16, 8, 3, 2, 1, True),
                                 (6, 3, 6, 16, 8, 3, 2, 2, True),
                                 (20, 3, 6, 16, 8, 3, 2, 2, True),
                                 (1, 3, 6, 4, 2, 3, 2, 1, True),
                                 (1, 3, 6, 128, 64, 3, 2, 1, True),
                                 (1, 3, 6, 512, 256, 3, 2, 1, True),
                                 (1, 1, 6, 16, 8, 3, 2, 2, True),
                                 (1, 3, 6, 16, 8, 5, 2, 2, True),
                                 (1, 3, 6, 16, 8, 3, 2, 4, True),
                                 (1, 3, 6, 16, 8, 5, 2, 1, True),
                                 (1, 3, 6, 16, 8, 5, 2, 3, True),
                                 # Without residual, stride=2
                                 (1, 3, 6, 16, 8, 3, 2, 1, False),
                                 (1, 3, 6, 16, 8, 5, 2, 2, False),
                             ])
    def test_encoder2d_local_refinement_forward_pass_shapes(self,
                                                            batch_size: int,
                                                            in_channels: int,
                                                            out_channels: int,
                                                            input_size: int,
                                                            expected_size: int,
                                                            kernel: IntOrTuple2d,
                                                            stride: IntOrTuple2d,
                                                            dilation: IntOrTuple2d,
                                                            use_residual: bool) -> None:
        """Output shape is correct for local_refinement mode across all parameter combinations."""
        x = randn(batch_size, in_channels, input_size, input_size).float()
        encoder = Encoder2D(in_channels=in_channels,
                            out_channels=out_channels,
                            conv_params=get_encoder_params_2d(kernel=kernel, stride=stride, dilation=dilation),
                            use_residual=use_residual,
                            apply_bias=False,
                            activation=ReLU,
                            layer_number=0,
                            residual_mode="local_refinement")
        assert encoder(x).shape == (batch_size, out_channels, expected_size, expected_size)

    # ------------------------------------------------------------------
    # Backward pass tests — separate per mode because he2 has no final ReLU
    # ------------------------------------------------------------------

    def test_encoder2d_local_refinement_backward_pass(self) -> None:
        """local_refinement backward pass: loss is finite, output is non-negative (post-activation)."""
        x = randn(2, 3, 16, 16).float()
        gt = randn(2, 6, 16, 16).float()
        encoder = Encoder2D(in_channels=3,
                            out_channels=6,
                            conv_params=get_encoder_params_2d(kernel=3, stride=1, dilation=1),
                            use_residual=True,
                            apply_bias=False,
                            activation=ReLU,
                            layer_number=0,
                            residual_mode="local_refinement")
        output = encoder(x)
        loss = sqrt(MSELoss()(output, gt) + 1e-8)
        loss.backward()

        assert output.shape == (2, 6, 16, 16)
        assert isinstance(loss.item(), float)
        assert loss.item() > 0
        assert isfinite(loss).item() is True
        # local_refinement applies activation inside conv_refine so output is non-negative
        from torch import all as tall
        assert tall(output >= 0.0)
        assert isfinite(output).all()

    def test_encoder2d_he2_backward_pass(self) -> None:
        """he2 backward pass: loss is finite and output is finite (no final ReLU so sign is unconstrained)."""
        x = randn(2, 3, 16, 16).float()
        gt = randn(2, 6, 16, 16).float()
        encoder = Encoder2D(in_channels=3,
                            out_channels=6,
                            conv_params=get_encoder_params_2d(kernel=3, stride=1, dilation=1),
                            use_residual=True,
                            apply_bias=False,
                            activation=ReLU,
                            layer_number=0,
                            residual_mode="he2")
        output = encoder(x)
        loss = sqrt(MSELoss()(output, gt) + 1e-8)
        loss.backward()

        assert output.shape == (2, 6, 16, 16)
        assert isinstance(loss.item(), float)
        assert loss.item() > 0
        assert isfinite(loss).item() is True
        assert isfinite(output).all()

    # ------------------------------------------------------------------
    # he2-specific structural tests
    # ------------------------------------------------------------------

    def test_encoder2d_he2_requires_use_residual(self) -> None:
        """he2 mode requires use_residual=True and raises ValueError otherwise."""
        with pytest.raises(ValueError, match="use_residual=False"):
            Encoder2D(in_channels=3,
                      out_channels=6,
                      conv_params=get_encoder_params_2d(kernel=3, stride=2, dilation=1),
                      use_residual=False,
                      apply_bias=False,
                      activation=ReLU,
                      layer_number=0,
                      residual_mode="he2")

    @pytest.mark.parametrize("in_channels, out_channels, stride", [
        (3, 6, 1),   # channel mismatch only
        (3, 6, 2),   # channel and spatial mismatch
        (8, 8, 2),   # spatial mismatch only
        (8, 16, 2),  # both
    ])
    def test_encoder2d_he2_shortcut_is_always_projection(self,
                                                         in_channels: int,
                                                         out_channels: int,
                                                         stride: int) -> None:
        """he2 shortcut is always a 1x1 Conv2dLayer projection (every encoder block changes shape)."""
        encoder = Encoder2D(in_channels=in_channels,
                            out_channels=out_channels,
                            conv_params=get_encoder_params_2d(kernel=3, stride=stride, dilation=1),
                            use_residual=True,
                            apply_bias=False,
                            activation=ReLU,
                            layer_number=0,
                            residual_mode="he2")
        assert isinstance(encoder.shortcut, Conv2dLayer)
        assert encoder.shortcut.conv2d.kernel_size == (1, 1)

    def test_encoder2d_he2_residual_gradient_flows_through_shortcut(self) -> None:
        """Projection shortcut weights receive non-zero gradient after backward, confirming the residual path is live."""
        encoder = Encoder2D(in_channels=3,
                            out_channels=8,
                            conv_params=get_encoder_params_2d(kernel=3, stride=2, dilation=1),
                            use_residual=True,
                            apply_bias=False,
                            activation=ReLU,
                            layer_number=0,
                            residual_mode="he2")

        x = randn(2, 3, 16, 16).float()
        gt = randn(2, 8, 8, 8).float()
        MSELoss()(encoder(x), gt).backward()

        grad = encoder.shortcut.conv2d.weight.grad
        assert grad is not None
        assert grad.abs().sum().item() > 0.0

    # ------------------------------------------------------------------
    # local_refinement-specific structural tests
    # ------------------------------------------------------------------

    def test_encoder2d_invalid_residual_mode_raises_in_forward(self) -> None:
        """forward() raises ValueError for an unrecognised residual_mode (guards the else branch)."""
        encoder = Encoder2D(in_channels=3,
                            out_channels=6,
                            conv_params=get_encoder_params_2d(kernel=3, stride=1, dilation=1),
                            use_residual=True,
                            apply_bias=False,
                            activation=ReLU,
                            layer_number=0,
                            residual_mode="he2")
        encoder.residual_mode = "invalid_mode"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="invalid_mode"):
            encoder(randn(1, 3, 8, 8).float())

    def test_encoder2d_local_refinement_use_residual_false(self) -> None:
        """local_refinement with use_residual=False produces the same shape as use_residual=True."""
        params = get_encoder_params_2d(kernel=3, stride=2, dilation=1)

        enc_with = Encoder2D(in_channels=3, out_channels=6, conv_params=params,
                             use_residual=True, apply_bias=False, activation=ReLU,
                             layer_number=0, residual_mode="local_refinement")
        enc_without = Encoder2D(in_channels=3, out_channels=6, conv_params=params,
                                use_residual=False, apply_bias=False, activation=ReLU,
                                layer_number=0, residual_mode="local_refinement")

        x = randn(1, 3, 16, 16).float()
        assert enc_with(x).shape == enc_without(x).shape
