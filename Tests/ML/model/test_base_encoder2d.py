import torch
import pytest
from SkiNet.ML.model.base_encoder2d import BaseEncoder2D
from SkiNet.ML.utils.sampling.encoder_sampling import PaddingMode


class TestBaseEncoder2D:
    """Unit tests for BaseEncoder2D class."""

    @pytest.mark.parametrize("batch_size, in_channels, out_channels, input_shape, kernel, stride, padding_mode, dilation, use_residual", [
        # all inputs are multiples of 2 because of the downsampling by factor of 2, to ensure compatibility across depths of the network
        # SAME padding tests
        (1, 3, 7, 16, 3, 1, PaddingMode.SAME, 1, True),
        # Vary batch
        (6, 3, 7, 16, 3, 1, PaddingMode.SAME, 1, True),
        (20, 3, 7, 16, 3, 1, PaddingMode.SAME, 1, True),
        # Vary input size
        (1, 3, 7, 8, 3, 1, PaddingMode.SAME, 1, True),
        (1, 3, 7, 128, 3, 1, PaddingMode.SAME, 1, True),
        (1, 3, 7, 512, 3, 1, PaddingMode.SAME, 1, True),
        # Vary channels
        (1, 1, 7, 16, 3, 1, PaddingMode.SAME, 1, True),
        (1, 3, 15, 16, 3, 1, PaddingMode.SAME, 1, True),
        # Vary kernel and dilation
        (1, 3, 7, 16, 3, 1, PaddingMode.SAME, 1, True), # s=1, k=2n+1, d=1; n=1,2,3,... integer padding as per exact formula
        (1, 3, 7, 16, 5, 1, PaddingMode.SAME, 1, True),
        (1, 3, 7, 16, 2, 1, PaddingMode.SAME, 1, True), # s=1, d=1; non-integer padding as per exact formula due to k=2n, but still works with PyTorch 'same' padding
        (1, 3, 7, 16, 4, 1, PaddingMode.SAME, 1, True),
        (1, 3, 7, 16, 3, 1, PaddingMode.SAME, 2, True), # s=1, k=any, d=2, integer padding as per exact formula
        (1, 3, 7, 16, 4, 1, PaddingMode.SAME, 2, True),
        (1, 3, 7, 16, 5, 1, PaddingMode.SAME, 2, True),
        (1, 3, 7, 16, 3, 1, PaddingMode.SAME, 3, True), # s=1, k=2n+1, d=3; n=1,2,3,... integer padding as per exact formula
        (1, 3, 7, 16, 5, 1, PaddingMode.SAME, 3, True),
        (1, 3, 7, 16, 2, 1, PaddingMode.SAME, 3, True), # s=1, d=3; non-integer padding as per exact formula due to k=2n, but still works with PyTorch 'same' padding
        (1, 3, 7, 16, 4, 1, PaddingMode.SAME, 3, True),
        # Without residual
        (1, 3, 7, 16, 3, 1, PaddingMode.SAME, 1, False), # s=1, k=2n+1, d=1; n=1,2,3,... integer padding as per exact formula
        (1, 3, 7, 16, 2, 1, PaddingMode.SAME, 1, False), # s=1, d=1; non-integer padding as per exact formula due to k=2n, but still works with PyTorch 'same' padding
        (1, 3, 7, 16, 3, 1, PaddingMode.SAME, 2, False), # s=1, k=any, d=2, integer padding as per exact formula
        (1, 3, 7, 16, 3, 1, PaddingMode.SAME, 3, False), # s=1, k=2n+1, d=3; n=1,2,3,... integer padding as per exact formula
        (1, 3, 7, 16, 2, 1, PaddingMode.SAME, 3, False), # s=1, d=3; non-integer padding as per exact formula due to k=2n, but still works with PyTorch 'same' padding
        #
        # VALID padding tests
        (1, 3, 7, 16, 3, 1, PaddingMode.VALID, 1, True),
        # Vary batch
        (6, 3, 7, 16, 3, 1, PaddingMode.VALID, 1, True),
        (20, 3, 7, 16, 3, 1, PaddingMode.VALID, 1, True),
        # Vary input size
        (4, 3, 7, 128, 3, 1, PaddingMode.VALID, 1, True),
        (4, 3, 7, 512, 3, 1, PaddingMode.VALID, 1, True),
        # Vary channels
        (1, 1, 7, 16, 3, 1, PaddingMode.VALID, 1, True),
        (1, 3, 15, 16, 3, 1, PaddingMode.VALID, 1, True),
        # Vary kernel and dilation
        (1, 3, 7, 16, 2, 1, PaddingMode.VALID, 1, True),
        (1, 3, 7, 16, 3, 1, PaddingMode.VALID, 1, True),
        (1, 3, 7, 16, 3, 1, PaddingMode.VALID, 2, True),
        (1, 3, 7, 16, 4, 1, PaddingMode.VALID, 2, True),
        (1, 3, 7, 16, 3, 1, PaddingMode.VALID, 3, True),
        (1, 3, 7, 16, 5, 1, PaddingMode.VALID, 3, True),
        # Without residual
        (1, 3, 7, 16, 3, 1, PaddingMode.VALID, 1, False),
        (1, 3, 7, 16, 3, 1, PaddingMode.VALID, 2, False),
        (1, 3, 7, 16, 3, 1, PaddingMode.VALID, 3, False),
        #
        # DOWNSAMPLING_FACTOR_2 tests
        (1, 3, 7, 16, 3, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, True),
        # Vary batch
        (6, 3, 7, 16, 3, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, True),
        (20, 3, 7, 16, 3, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, True),
        # Vary input size
        (1, 3, 7, 4, 2, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, True), # 2x2 output
        (1, 3, 7, 4, 3, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, True), # 2x2 output
        (1, 3, 7, 4, 3, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 2, True), # 2x2 output
        (1, 3, 7, 6, 3, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 2, True), # 3x3 output
        (1, 3, 7, 8, 3, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, True),
        (1, 3, 7, 128, 3, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, True),
        (1, 3, 7, 512, 3, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, True),
        # Vary channels
        (1, 1, 7, 16, 3, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, True),
        (1, 3, 15, 16, 3, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, True),
        # Vary kernel and dilation
        (1, 3, 7, 16, 4, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, True), # s=2, k=2n, d=1; n=1,2,3,... integer padding as per exact formula
        (1, 3, 7, 16, 6, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, True),
        (1, 3, 7, 16, 3, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, True),   # s=1, d=1; non-integer padding as per exact formula due to k=2n+1, but still works with PyTorch 'same' padding
        (1, 3, 7, 16, 5, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, True),
        (1, 3, 7, 16, 3, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 2, True), #   s=1, non-integer padding as per exact formula due to d=2n, but still works with PyTorch 'same' padding
        (1, 3, 7, 16, 4, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 2, True),
        (1, 3, 7, 16, 4, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 3, True),  # s=2, k=2n, d=3; n=1,2,3,... integer padding as per exact formula
        (1, 3, 7, 16, 6, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 3, True),
        (1, 3, 7, 16, 3, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 3, True),  # s=1, d=3; non-integer padding as per exact formula due to k=2n+1, but still works with PyTorch 'same' padding
        (1, 3, 7, 16, 5, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 3, True),
        # vary residual
        (1, 3, 7, 16, 4, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, False), # s=2, k=2n, d=1; n=1,2,3,... integer padding as per exact formula
        (1, 3, 7, 16, 6, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, False),
        (1, 3, 7, 16, 3, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, False),   # s=1, d=1; non-integer padding as per exact formula due to k=2n+1, but still works with PyTorch 'same' padding
        (1, 3, 7, 16, 5, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, False),
        (1, 3, 7, 16, 3, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 2, False),   # s=1, non-integer padding as per exact formula due to d=2n, but still works with PyTorch 'same' padding
        (1, 3, 7, 16, 4, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 2, False),
        (1, 3, 7, 16, 4, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 3, False),  # s=2, k=2n, d=3; n=1,2,3,... integer padding as per exact formula
        (1, 3, 7, 16, 6, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 3, False),
        (1, 3, 7, 16, 3, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 3, False),   # s=1, d=3; non-integer padding as per exact formula due to k=2n+1, but still works with PyTorch 'same' padding
        (1, 3, 7, 16, 5, 2, PaddingMode.DOWNSAMPLING_FACTOR_2, 3, False),
   ])
    def test_forward_pass_shapes(self, batch_size, in_channels, out_channels, input_shape, kernel, stride, padding_mode, dilation, use_residual):
        """Test forward pass with various parameters and exact output shape verification."""
        encoder = BaseEncoder2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=kernel,
            stride=stride,
            padding_mode=padding_mode,
            dilation=dilation,
            use_residual=use_residual
        )

        input_height = input_width = input_shape
        x = torch.randn(batch_size, in_channels, input_height, input_width)
        output = encoder(x)

        if padding_mode == PaddingMode.SAME:
            expected_height = input_height
            expected_width = input_width
        elif padding_mode == PaddingMode.VALID:
            # after first conv:
            first_height = (input_height - kernel - (kernel - 1) * (dilation - 1)) // stride + 1
            first_width =  (input_width - kernel - (kernel - 1) * (dilation - 1)) // stride + 1
            # after second conv always SAME padding
            expected_height = first_height
            expected_width = first_width
        elif padding_mode == PaddingMode.DOWNSAMPLING_FACTOR_2:
            expected_height = input_height // 2
            expected_width = input_width // 2
        # batch_size and out_channels are conserved
        expected_shape = (batch_size, out_channels, expected_height, expected_width)
        assert output.shape == expected_shape
