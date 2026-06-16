import pytest
from torch import Tensor, nn, randn

from SkiNet.ML.model.blocks.decoder2d import Decoder2D
from SkiNet.ML.utils.sampling.decoder_sampling import get_decoder_params_2d
from SkiNet.ML.utils.sampling.encoder_sampling import get_encoder_params_2d


@pytest.mark.parametrize("batch, out_channels, in_channels, kernel, stride, dilation, input_size, expected_size",
                         [  # Vary batch
                             (4, 6, 12, 3, 2, 1, 8, 16),
                             # Vary input size
                             (2, 6, 12, 3, 2, 1, 16, 32),
                             # Vary channels
                             (2, 3, 6, 3, 2, 1, 16, 32),
                             # Vary kernel and dilation, stride = 2, i.e. upsampling = 2 are fixed
                             (2, 6, 12, 3, 2, 1, 8, 16),  # k=2n+1, d=2n+1
                             (2, 6, 12, 5, 2, 1, 32, 64),
                             (2, 6, 12, 3, 2, 2, 8, 16),  # d=2n
                             (2, 6, 12, 7, 2, 2, 32, 64)
                         ])
def test_forward_and_output_shape(batch: int,
                                  in_channels: int,
                                  out_channels: int,
                                  kernel: int,
                                  stride: int,
                                  dilation: int,
                                  input_size: int,
                                  expected_size: int) -> None:
    """
    Test forward pass with various parameters and exact output shape verification.
    """
    params = get_encoder_params_2d(kernel=kernel, stride=stride, dilation=dilation)

    layer = Decoder2D(in_channels=in_channels,
                      out_channels=out_channels,
                      decoder_params=get_decoder_params_2d(params),
                      activation=nn.ReLU,
                      layer_number=1)

    input = randn(batch, in_channels, input_size, input_size).float()

    output = layer(input)

    assert isinstance(output, Tensor)
    assert output.shape == (batch, out_channels, expected_size, expected_size)
