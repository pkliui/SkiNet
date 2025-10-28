import torch
import pytest
from SkiNet.ML.model.decoder2d import Decoder2D
from SkiNet.ML.utils.sampling.encoder_sampling import PaddingMode, get_padding
from SkiNet.ML.utils.sampling.decoder_sampling import EncoderSpec


@pytest.mark.parametrize("batch_size, in_channels, out_channels, input_shape, kernel, stride, dilation, padding_mode", [
    (2, 6, 3, (20, 20), (4, 4), (2, 2), (1,1), PaddingMode.DOWNSAMPLING_FACTOR_2),
    (2, 6, 3, (512, 512), (6, 6), (2, 2), (1,1), PaddingMode.DOWNSAMPLING_FACTOR_2),
    (2, 6, 3, (512, 512), (8, 8), (2, 2), (1,1), PaddingMode.DOWNSAMPLING_FACTOR_2)
])

def test_decoder2d_returns_expected_shapes(batch_size, in_channels, out_channels, input_shape, kernel, stride, dilation, padding_mode) -> None:
    """
    Test Decoder2D: output spatial dimensions should be input dimensions scaled by stride.
    """
    padding = get_padding(kernel=kernel,
                        dilation=dilation,
                        padding_mode=padding_mode,
                        num_dims=2,
                        stride=stride)

    encoder_spec = EncoderSpec(kernel=kernel,
        stride=stride,
        padding=padding)

    decoder = Decoder2D(in_channels=in_channels,
        out_channels=out_channels,
        encoder_spec=encoder_spec,
        activation=torch.nn.ReLU)

    # Upsample input
    x = torch.randn(batch_size, in_channels, *input_shape)
    out = decoder(x)

    # Assert output is input scaled by stride
    assert out.shape == (batch_size, out_channels, input_shape[0] * stride[0], input_shape[1] * stride[1])