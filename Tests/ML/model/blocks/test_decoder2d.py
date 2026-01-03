import pytest
import torch
import torch.nn as nn
from torch import Tensor

from SkiNet.ML.model.blocks.decoder2d import Decoder2D


@pytest.mark.parametrize(
    "batch,in_channels,out_channels,kernel,stride,input_size,expected_size",
    [
        # stride = 2 → upsample
        (2, 8, 4, 3, 2, 10, 20),
        (1, 16, 8, 4, 2, 5, 10),
        (4, 3, 6, 5, 2, 7, 14),

        # stride = 1 → same size
        (2, 8, 4, 3, 1, 10, 10),
        (1, 16, 8, 5, 1, 7, 7),
    ],
)
def test_decoder2d_forward_and_shape(batch: int,
                                     in_channels: int,
                                     out_channels: int,
                                     kernel: int,
                                     stride: int,
                                     input_size: int,
                                     expected_size: int) -> None:
    decoder = Decoder2D(in_channels=in_channels,
                        out_channels=out_channels,
                        decoding_kernel_size=kernel,
                        decoding_stride=stride,
                        activation=nn.ReLU)

    x = torch.rand(batch, in_channels, input_size, input_size)
    y = decoder(x)

    assert isinstance(y, Tensor)
    assert torch.all(y >= 0)
    assert y.shape == (batch, out_channels, expected_size, expected_size)
