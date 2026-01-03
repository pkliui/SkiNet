import pytest
import torch

from SkiNet.ML.model.blocks.encoder2d import Encoder2D
from SkiNet.ML.utils.typing_utils import IntOrTuple2d


class TestEncoder2D:
    """
    Unit tests for Encoder2D class
    """
    @pytest.mark.parametrize("batch_size, in_channels, out_channels, input_shape, kernel, stride, dilation, use_residual", [
        # even kernels and odd dilations result in fractional padding and output tensor dimensions do not match exteted ones
        # (see encoder_sampling for details)

        # SAME padding tests
        (1, 3, 6, 16, 3, 1, 1, True),
        # Vary batch
        (6, 3, 6, 16, 3, 1, 1, True),
        (20, 3, 6, 16, 3, 1, 1, True),
        # Vary input size
        (1, 3, 6, 8, 3, 1, 1, True),
        (1, 3, 6, 128, 3, 1, 1, True),
        (1, 3, 6, 512, 3, 1, 1, True),
        # Vary channels
        (1, 1, 2, 16, 3, 1, 1, True),
        (1, 16, 32, 16, 3, 1, 1, True),
        # Vary kernel and dilation
        (1, 3, 3, 16, 3, 1, 1, True),  # s=1, k=2n+1, d=2n+1;
        (1, 3, 3, 16, 5, 1, 1, True),
        (1, 3, 3, 16, 3, 1, 3, True),
        (1, 3, 3, 16, 5, 1, 3, True),
        (1, 3, 3, 16, 2, 1, 2, True),  # s=1, k=2n, d=2n;
        (1, 3, 3, 16, 4, 1, 2, True),
        (1, 3, 3, 16, 2, 1, 4, True),
        (1, 3, 3, 16, 4, 1, 4, True),
        (1, 3, 6, 32, 3, 1, 2, True),  # s=1, k=2n+1, d=2n
        (1, 3, 6, 32, 5, 1, 2, True),
        (1, 3, 6, 32, 5, 1, 4, True),
        # # Without residual
        (1, 3, 3, 16, 3, 1, 1, False),  # s=1, k=2n+1, d=2n+1;
        (1, 3, 3, 16, 5, 1, 1, False),
        (1, 3, 3, 16, 4, 1, 2, False),  # s=1, k=2n, d=2n;
        (1, 3, 3, 16, 6, 1, 2, False),
        (1, 3, 3, 16, 4, 1, 2, False),

        # DOWNSAMPLING_FACTOR_2 tests
        (1, 3, 6, 16, 3, 2, 1, True),
        # Vary batch
        (6, 3, 6, 16, 4, 2, 2, True),  # s=2, k=2n, d=2n
        (20, 3, 6, 16, 4, 2, 2, True),
        # Vary input size
        (1, 3, 6, 4, 4, 2, 2, True),  # s=2, k=2n, d=2n
        (1, 3, 6, 128, 4, 2, 2, True),
        (1, 3, 6, 512, 4, 2, 2, True),
        # Vary channels
        (1, 1, 6, 16, 4, 2, 2, True),
        (1, 3, 6, 16, 4, 2, 2, True),
        # Vary kernel and dilation
        (1, 3, 6, 16, 3, 2, 1, True),  # s=2, k=2n+1, d=2n+1
        (1, 3, 6, 16, 5, 2, 1, True),
        (1, 3, 6, 16, 3, 2, 3, True),
        (1, 3, 6, 16, 5, 2, 3, True),
        (1, 3, 6, 16, 4, 2, 2, True),  # s=2, k=2n, d=2n
        (1, 3, 6, 16, 6, 2, 2, True),
        (1, 3, 6, 32, 4, 2, 4, True),
        (1, 3, 6, 32, 6, 2, 4, True),
        (1, 3, 6, 32, 3, 2, 2, True),  # s=2, k=2n+1, d=2n
        (1, 3, 6, 32, 5, 2, 2, True),
        (1, 3, 6, 32, 5, 2, 4, True),

        # vary residual
        (1, 3, 6, 16, 3, 2, 1, False),  # s=2, k=2n+1, d=2n+1
        (1, 3, 6, 16, 4, 2, 2, False),  # s=2, k=2n, d=2n
    ])
    def test_forward_pass_shapes(self,
                                 batch_size: int,
                                 in_channels: int,
                                 out_channels: int,
                                 input_shape: int,
                                 kernel: IntOrTuple2d,
                                 stride: IntOrTuple2d,
                                 dilation: IntOrTuple2d,
                                 use_residual: bool) -> None:
        """Test forward pass with various parameters and exact output shape verification."""
        encoder = Encoder2D(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel=kernel,
                            stride=stride,
                            dilation=dilation,
                            use_residual=use_residual)

        input_height = input_width = input_shape
        x = torch.randn(batch_size, in_channels, input_height, input_width)
        output = encoder(x)

        if stride == 1:
            expected_height = input_height
            expected_width = input_width
        elif stride == 2:
            expected_height = input_height // 2
            expected_width = input_width // 2
        # batch_size and out_channels are conserved
        expected_shape = (batch_size, out_channels, expected_height, expected_width)
        assert output.shape == expected_shape
