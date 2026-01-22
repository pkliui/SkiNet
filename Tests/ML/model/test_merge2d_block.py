import pytest
from torch import nn, randn

from SkiNet.ML.model.blocks.decoder2d import Decoder2D
from SkiNet.ML.model.blocks.merge2d_block import Merge2DBlock
from SkiNet.ML.utils.sampling.decoder_sampling import get_decoder_params_2d
from SkiNet.ML.utils.sampling.encoder_sampling import get_encoder_params_2d
from SkiNet.ML.utils.typing_utils import TupleOfInt2d


@pytest.mark.parametrize("batch_size, in_channels, out_channels, input_shape, kernel, stride, dilation", [
    (2, 8, 4, (32, 32), (3, 3), (2, 2), (1, 1)),
])
def test_merge2d_block_forward(batch_size: int,
                               in_channels: int,
                               out_channels: int,
                               input_shape: TupleOfInt2d,
                               kernel: TupleOfInt2d,
                               stride: TupleOfInt2d,
                               dilation: TupleOfInt2d) -> None:
    """
    Test the output of the Merge2DBlock has expected dimensions after merging decoder output and skip connection.
    """
    params = get_encoder_params_2d(kernel=kernel, stride=stride, dilation=dilation)

    decoder = Decoder2D(in_channels=in_channels,
                        out_channels=out_channels,
                        decoder_params=get_decoder_params_2d(params),
                        activation=nn.ReLU,
                        layer_number=1)

    merge_block = Merge2DBlock(layer_number=0,
                               in_channels=out_channels,
                               out_channels=out_channels,
                               conv_params=params,
                               activation=nn.ReLU)

    # input_to_decoder will be upsampled by the decoder to the dimensions of the skip connection
    input_to_decoder = randn(batch_size, in_channels, input_shape[0], input_shape[1])
    skip_connection = randn(batch_size, out_channels, input_shape[0]*stride[0], input_shape[1]*stride[1])

    decoder_output = decoder(input_to_decoder)
    assert skip_connection.shape == decoder_output.shape

    # now merge skip connection and decoder's output
    output_of_merge_block = merge_block(decoder_output, skip_connection)
    # batch size and the number of channels are conserved
    # the shape is input shape to the decoder scaled by stride
    assert output_of_merge_block.shape == (batch_size, out_channels, input_shape[0]*stride[0], input_shape[1]*stride[1])
