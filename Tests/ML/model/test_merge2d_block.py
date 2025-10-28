import pytest
import torch
from SkiNet.ML.model.decoder2d import Decoder2D
from SkiNet.ML.model.merge2d_block import Merge2DBlock
from SkiNet.ML.utils.sampling.decoder_sampling import EncoderSpec
from SkiNet.ML.utils.sampling.encoder_sampling import PaddingMode, get_padding

@pytest.mark.parametrize("batch_size, in_channels, out_channels, input_shape, kernel, stride, dilation, padding_mode", [
    (2, 8, 4, (32, 32), (4, 4), (2, 2), (1, 1), PaddingMode.DOWNSAMPLING_FACTOR_2),
])
def test_merge2d_block_forward(batch_size, in_channels, out_channels, input_shape, kernel, stride, dilation, padding_mode):
    """
    Test the output of the Merge2DBlock has expected dimensions after merging decoder output and skip connection.
    """
    padding = get_padding(kernel=kernel,
                                   dilation=dilation,
                                   padding_mode=padding_mode,
                                   num_dims=2,
                                   stride=stride)

    encoder_spec = EncoderSpec(kernel=kernel,
                               stride=stride,
                               padding=padding)

    decoder = Decoder2D(in_channels,
                        out_channels,
                        encoder_spec=encoder_spec,
                        activation=torch.nn.ReLU)

    merge_block = Merge2DBlock(
        in_channels=out_channels,
        out_channels=out_channels,
        kernel=kernel,
        apply_batchnorm=True,
        apply_bias=False,
        activation=torch.nn.ReLU
    )

    # shall be upsampled by decoder
    input_to_decoder = torch.randn(batch_size, in_channels, input_shape[0], input_shape[1])

    # shall have the same dimensions as the decoder output
    skip_connection = torch.randn(batch_size, out_channels, input_shape[0]*stride[0], input_shape[1]*stride[1])

    output_of_merge_block = merge_block(decoder(input_to_decoder), skip_connection)

    # batch size stays the same, the number of channels is output_channels
    # the shape is input shape to the decoder scaled by stride
    assert output_of_merge_block.shape == (batch_size, out_channels, input_shape[0]*stride[0], input_shape[1]*stride[1])
