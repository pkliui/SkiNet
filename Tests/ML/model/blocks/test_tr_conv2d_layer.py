import pytest
import torch
import torch.nn as nn
from torch import Tensor

from SkiNet.ML.model.blocks.tr_conv2d_layer import TrConv2dLayer


@pytest.mark.parametrize("batch, out_channels, in_channels, kernel, stride, input_size, expected_size",
                         [   # stride = 2, downsampling = 2
                             (2, 6, 12, 3, 2, 10, 20),  # stride=2, k=2n+1, op=1
                             (2, 6, 12, 4, 2, 10, 20),  # stride=2, k=2n,  op=0
                             (2, 6, 12, 5, 2, 10, 20),  # stride=2, k=2n+1,  op=1
                             (2, 6, 12, 6, 2, 10, 20),  # stride=2, k=2n,  op=0
                             # stride = 1, same size
                             (2, 6, 12, 3, 1, 10, 10),  # stride=1, k=2n+1, op=0
                             (2, 6, 12, 5, 1, 10, 10),  # stride=1, k=2n+1, op=0
                         ])
def test_forward_and_output_shape(batch: int,
                                  in_channels: int,
                                  out_channels: int,
                                  kernel: int,
                                  stride: int,
                                  input_size: int,
                                  expected_size: int) -> None:

    layer = TrConv2dLayer(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel,
                          stride=stride,
                          activation=nn.ReLU)

    input = torch.rand(batch, in_channels, input_size, input_size).float()

    output = layer(input)

    assert isinstance(output, Tensor)
    assert output.shape == (batch, out_channels, expected_size, expected_size)
