import pytest
from torch import all, isfinite, randn, sqrt
from torch.nn import MSELoss, ReLU

from SkiNet.ML.model.blocks.encoder2d import Encoder2D
from SkiNet.ML.utils.sampling.encoder_sampling import get_encoder_params_2d
from SkiNet.ML.utils.typing_utils import IntOrTuple2d


class TestEncoder2D:
    """
    Unit tests for Encoder2D class
    """
    @pytest.mark.parametrize("batch_size, in_channels, out_channels, input_size, expected_size, kernel, stride, dilation, use_residual",
                             [
                                 # SAME padding tests
                                 (1, 3, 6, 16, 16, 3, 1, 1, True),
                                 # Vary batch
                                 (6, 3, 6, 16, 16, 3, 1, 1, True),
                                 (20, 3, 6, 16, 16, 3, 1, 1, True),
                                 # Vary input size
                                 (1, 3, 6, 128, 128, 3, 1, 1, True),
                                 (1, 3, 6, 512, 512, 3, 1, 1, True),
                                 # Vary channels
                                 (1, 1, 2, 16, 16, 3, 1, 1, True),
                                 (1, 16, 32, 16, 16, 3, 1, 1, True),
                                 # Vary kernel and dilation
                                 (1, 3, 3, 16, 16, 3, 1, 1, True),  # s=1, k=2n+1, d=2n+1;
                                 (1, 3, 3, 16, 16, 5, 1, 1, True),
                                 (1, 3, 3, 16, 16, 3, 1, 3, True),
                                 (1, 3, 3, 16, 16, 5, 1, 3, True),
                                 (1, 3, 6, 32, 32, 3, 1, 2, True),  # s=1, k=2n+1, d=2n
                                 (1, 3, 6, 32, 32, 3, 1, 4, True),
                                 (1, 3, 6, 32, 32, 5, 1, 2, True),
                                 # # Without residual
                                 (1, 3, 3, 16, 16, 3, 1, 1, False),  # s=1, k=2n+1, d=2n+1
                                 (1, 3, 3, 16, 16, 5, 1, 2, False),  # s=1, k=2n+1, d=2n
                                 # DOWNSAMPLING_FACTOR_2 tests
                                 (1, 3, 6, 16, 8, 3, 2, 1, True),
                                 # Vary batch
                                 (6, 3, 6, 16, 8, 3, 2, 2, True),  # s=2, k=2n, d=2n
                                 (20, 3, 6, 16, 8, 3, 2, 2, True),
                                 # Vary input size
                                 (1, 3, 6, 4, 2, 3, 2, 1, True),  # s=2, k=2n+1
                                 (1, 3, 6, 128, 64, 3, 2, 1, True),
                                 (1, 3, 6, 512, 256, 3, 2, 1, True),
                                 # Vary channels
                                 (1, 1, 6, 16, 8, 3, 2, 2, True),
                                 (1, 3, 6, 16, 8, 3, 2, 2, True),
                                 # Vary kernel and dilation
                                 (1, 3, 6, 16, 8, 3, 2, 2, True),  # s=2, k=2n+1, d=2n
                                 (1, 3, 6, 16, 8, 5, 2, 2, True),
                                 (1, 3, 6, 16, 8, 3, 2, 4, True),
                                 (1, 3, 6, 16, 8, 3, 2, 1, True),  # s=2, k=2n+1, d=2n+1
                                 (1, 3, 6, 16, 8, 5, 2, 1, True),
                                 (1, 3, 6, 16, 8, 5, 2, 3, True),
                                 # vary residual
                                 (1, 3, 6, 16, 8, 3, 2, 1, False),  # s=2, k=2n+1, d=2n+1
                                 (1, 3, 6, 16, 8, 5, 2, 2, False)  # s=2, k=2n+1, d=2n
                             ])
    def test_encoder2d_forward_pass_shapes(self,
                                           batch_size: int,
                                           in_channels: int,
                                           out_channels: int,
                                           input_size: int,
                                           expected_size: int,
                                           kernel: IntOrTuple2d,
                                           stride: IntOrTuple2d,
                                           dilation: IntOrTuple2d,
                                           use_residual: bool) -> None:
        """
        Test forward pass with various parameters and exact output shape verification.
        """

        input = randn(batch_size, in_channels, input_size, input_size).float()

        encoder = Encoder2D(in_channels=in_channels,
                            out_channels=out_channels,
                            conv_params=get_encoder_params_2d(kernel=kernel, stride=stride, dilation=dilation),
                            use_residual=use_residual,
                            apply_bias=False,
                            apply_batchnorm=True,
                            activation=ReLU,
                            layer_number=0)
        output = encoder(input)
        assert output.shape == (batch_size, out_channels, expected_size, expected_size)

    def test_encoder2d_backward_pass(self) -> None:

        batch_size = 2
        in_channels = 3
        out_channels = 6
        input_size = 16
        expected_size = input_size
        kernel = 3
        stride = 1
        dilation = 1

        input = randn(batch_size, in_channels, input_size, input_size).float()
        ground_truth = randn(batch_size, out_channels, expected_size, expected_size).float()

        encoder = Encoder2D(in_channels=in_channels,
                            out_channels=out_channels,
                            conv_params=get_encoder_params_2d(kernel=kernel, stride=stride, dilation=dilation),
                            use_residual=True,
                            apply_bias=False,
                            apply_batchnorm=True,
                            activation=ReLU,
                            layer_number=0,)
        output = encoder(input)
        loss = sqrt(MSELoss()(output, ground_truth)+1e-8)

        # Check backward pass
        loss.backward()

        # Check output size
        assert output.shape == (batch_size, out_channels, expected_size, expected_size)

        # Loss is a positive finite value
        assert isinstance(loss.item(), float)
        assert loss.item() > 0
        assert isfinite(loss).item() is True

        # Positive-valued output after ReLU
        assert all(output >= 0.0)
        assert isfinite(output).all()
