import torch
import pytest
from SkiNet.ML.model.blocks.encoder2d import Encoder2D
from SkiNet.ML.utils.sampling.encoder_sampling import PaddingMode


class TestEncoder2D:
    """
    Unit tests for Encoder2D class

    SAME padding - preserve the size of the input after convolution
    Encoder2D is using PyTorch 'same' padding.

    If no pytorch's 'same' padding is used, it can be computed using formula
    padding = (kernel-1)*dilation/2,
    where kernel and dilation sizes are constrained to the following values,
    ensuring that padding is an integer:
    s=1, k=2n+1, d=1; n=1,2,3,...
    s=1, k=n, d=2; n=1,2,3,...


    DOWNSAMPLING_FACTOR_2 padding
    Encoder2D is using p=(k - 1) * d // 2 formula to account for floor division.

    In general, padding is computed  using exact formula
    padding = ((kernel-1)*dilation -1)/2,
    where dilation and kernel sizes are constrained to have the following values,
    ensuring that the resulting padding is an integer:
    s=2, k=2n, d=2n+1; n=1,2,3,...

    """

    @pytest.mark.parametrize("batch_size, in_channels, out_channels, input_shape, kernel, stride, padding_mode, dilation, use_residual", [
        # all inputs are multiples of 2 because of the downsampling by factor of 2, to ensure compatibility across depths of the network

        # SAME padding tests
        (1, 3, 6, 16, 3, None, PaddingMode.SAME, 1, True),
        # Vary batch
        (6, 3, 6, 16, 3, None, PaddingMode.SAME, 1, True),
        (20, 3, 6, 16, 3, None, PaddingMode.SAME, 1, True),
        # Vary input size
        (1, 3, 6, 8, 3, None, PaddingMode.SAME, 1, True),
        (1, 3, 6, 128, 3, None, PaddingMode.SAME, 1, True),
        (1, 3, 6, 512, 3, None, PaddingMode.SAME, 1, True),
        # Vary channels
        (1, 1, 2, 16, 3, None, PaddingMode.SAME, 1, True),
        (1, 16, 32, 16, 3, None, PaddingMode.SAME, 1, True),
        # Vary kernel and dilation
        (1, 3, 3, 16, 3, None, PaddingMode.SAME, 1, True), # s=1, k=2n+1, d=1; n=1,2,3,... integer padding as per exact formula
        (1, 3, 3, 16, 5, None, PaddingMode.SAME, 1, True),
        (1, 3, 3, 16, 2, None, PaddingMode.SAME, 1, True), # s=1, d=1; non-integer padding as per exact formula due to k=2n, but still works with PyTorch 'same' padding
        (1, 3, 3, 16, 4, None, PaddingMode.SAME, 1, True),
        (1, 3, 3, 16, 3, None, PaddingMode.SAME, 2, True), # s=1, k=any, d=2, integer padding as per exact formula
        (1, 3, 3, 16, 4, None, PaddingMode.SAME, 2, True),
        (1, 3, 3, 16, 5, None, PaddingMode.SAME, 2, True),
        (1, 3, 3, 16, 6, None, PaddingMode.SAME, 2, True),
        (1, 3, 3, 16, 3, None, PaddingMode.SAME, 3, True), # s=1, k=2n+1, d=3; n=1,2,3,... integer padding as per exact formula
        (1, 3, 3, 16, 5, None, PaddingMode.SAME, 3, True),
        (1, 3, 3, 16, 2, None, PaddingMode.SAME, 3, True), # s=1, d=3; non-integer padding as per exact formula due to k=2n, but still works with PyTorch 'same' padding
        (1, 3, 3, 16, 4, None, PaddingMode.SAME, 3, True),
        # Without residual
        (1, 3, 3, 16, 3, None, PaddingMode.SAME, 1, False), # s=1, k=2n+1, d=1; n=1,2,3,... integer padding as per exact formula
        (1, 3, 3, 16, 2, None, PaddingMode.SAME, 1, False), # s=1, d=1; non-integer padding as per exact formula due to k=2n, but still works with PyTorch 'same' padding
        (1, 3, 3, 16, 3, None, PaddingMode.SAME, 2, False), # s=1, k=any, d=2, integer padding as per exact formula
        (1, 3, 3, 16, 3, None, PaddingMode.SAME, 3, False), # s=1, k=2n+1, d=3; n=1,2,3,... integer padding as per exact formula
        (1, 3, 3, 16, 2, None, PaddingMode.SAME, 3, False), # s=1, d=3; non-integer padding as per exact formula due to k=2n, but still works with PyTorch 'same' padding

        # DOWNSAMPLING_FACTOR_2 tests
        (1, 3, 6, 16, 4, None, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, True),
        # Vary batch
        (6, 3, 6, 16, 4, None, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, True),
        (20, 3, 6, 16, 4, None, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, True),
        # Vary input size
        (1, 3, 6, 4, 2, None, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, True), # 2x2 output
        (1, 3, 6, 4, 4, None, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, True), # 2x2 output
        (1, 3, 6, 4, 4, None, PaddingMode.DOWNSAMPLING_FACTOR_2, 2, True), # 2x2 output
        (1, 3, 6, 6, 4, None, PaddingMode.DOWNSAMPLING_FACTOR_2, 2, True), # 3x3 output
        (1, 3, 6, 8, 4, None, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, True),
        (1, 3, 6, 128, 4, None, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, True),
        (1, 3, 6, 512, 4, None, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, True),
        # (i + 2p - k) mod s = 0 cases to ensure exact downsampling by factor of 2, p = d*(k-1)//2
        (1, 3, 6, 16, 4, None, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, True), # 16x16 input, 8x8 output, kernel=4, d=1, p = 1
        (1, 3, 6, 16, 4, None, PaddingMode.DOWNSAMPLING_FACTOR_2, 2, True), # 16x16 input, 8x8 output, kernel=4,  d=2, p = 3
        (1, 3, 6, 16, 4, None, PaddingMode.DOWNSAMPLING_FACTOR_2, 3, True), # 16x16 input, 8x8 output, kernel=4,  d=3, p = 4
        # Vary channels
        (1, 1, 6, 16, 4, None, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, True),
        (1, 3, 6, 16, 4, None, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, True),
        # Vary kernel and dilation
        (1, 3, 6, 16, 4, None, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, True), # s=2, k=2n, d=1; n=1,2,3,... integer padding as per exact formula
        (1, 3, 6, 16, 6, None, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, True),
        (1, 3, 6, 16, 4, None, PaddingMode.DOWNSAMPLING_FACTOR_2, 2, True), # s=2, k=2n, d=2, still works in torch.nn.conv2d for even inputs
        (1, 3, 6, 16, 6, None, PaddingMode.DOWNSAMPLING_FACTOR_2, 2, True),
        (1, 3, 6, 32, 4, None, PaddingMode.DOWNSAMPLING_FACTOR_2, 3, True),  # s=2, k=2n, d=3; n=1,2,3,... integer padding as per exact formula
        (1, 3, 6, 32, 6, None, PaddingMode.DOWNSAMPLING_FACTOR_2, 3, True),
        (1, 3, 6, 32, 4, None, PaddingMode.DOWNSAMPLING_FACTOR_2, 4, True), # s=2, k=2n, d=4, still works in torch.nn.conv2d for even inputs
        (1, 3, 6, 32, 6, None, PaddingMode.DOWNSAMPLING_FACTOR_2, 4, True),
        # vary residual
        (1, 3, 6, 16, 4, None, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, False), # s=2, k=2n, d=1; n=1,2,3,... integer padding as per exact formula
        (1, 3, 6, 16, 6, None, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, False),
        (1, 3, 6, 16, 4, None, PaddingMode.DOWNSAMPLING_FACTOR_2, 2, False), # s=2, k=2n, d=2, still works in torch.nn.conv2d for even inputs
        (1, 3, 6, 16, 6, None, PaddingMode.DOWNSAMPLING_FACTOR_2, 2, False),
        (1, 3, 6, 16, 4, None, PaddingMode.DOWNSAMPLING_FACTOR_2, 3, False), # s=2, k=2n, d=3, n=1,2,3,... integer padding as per exact formula
        (1, 3, 6, 16, 6, None, PaddingMode.DOWNSAMPLING_FACTOR_2, 3, False)
   ])
    def test_forward_pass_shapes(self, batch_size, in_channels, out_channels, input_shape, kernel, stride, padding_mode, dilation, use_residual):
        """Test forward pass with various parameters and exact output shape verification."""
        encoder = Encoder2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=kernel,
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
        elif padding_mode == PaddingMode.DOWNSAMPLING_FACTOR_2:
            expected_height = input_height // 2
            expected_width = input_width // 2
        # batch_size and out_channels are conserved
        expected_shape = (batch_size, out_channels, expected_height, expected_width)
        assert output.shape == expected_shape
