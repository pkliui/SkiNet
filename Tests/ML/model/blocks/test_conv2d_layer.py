import numpy as np
import pytest
import torch
from torch.nn import MSELoss

from SkiNet.ML.model.blocks.conv2d_layer import Conv2dLayer


@pytest.mark.parametrize("batch_size, in_channels, out_channels, kernel, stride, dilation, apply_bias, apply_batchnorm, activation, input_size, expected_size",
                         [
                             (2, 3, 6, 3, 1, 1, False, True, torch.nn.ReLU, 10, 10),  # stride=1
                             (2, 6, 12, 3, 2, 1, False, True, torch.nn.ReLU, 20, 10),  # stride=2
                             (2, 6, 12, 4, 2, 2, False, True, torch.nn.ReLU, 10, 5),  # stride=2
                             (2, 6, 12, 6, 2, 2, False, True, torch.nn.ReLU, 20, 10),  # stride=2

                         ])
def test_basic_layer_forward_and_backward_pass(batch_size: int,
                                               in_channels: int,
                                               out_channels: int,
                                               kernel: int,
                                               stride: int,
                                               dilation: int,
                                               apply_bias: bool,
                                               apply_batchnorm: bool,
                                               activation: torch.nn.ReLU,
                                               input_size: int,
                                               expected_size: int) -> None:
    """
    Test forward and backward passes in Conv2dLayer layer
    """
    input = torch.rand(batch_size, in_channels, input_size, input_size).float()
    ground_truth = torch.rand(batch_size, out_channels, expected_size, expected_size).float()

    layer = Conv2dLayer(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel=kernel,
                        stride=stride,
                        dilation=dilation,
                        apply_bias=apply_bias,
                        apply_batchnorm=apply_batchnorm,
                        activation=activation)

    output = layer(input)
    loss_function = MSELoss()
    loss = torch.sqrt(loss_function(output, ground_truth))
    loss.backward()

    # Loss is a positive finite value
    assert isinstance(loss.item(), float)
    assert loss.item() > 0
    assert not np.isnan(loss.detach().numpy()).any()
    assert not np.isinf(loss.detach().numpy()).any()

    # Positive-valued output after ReLU
    assert np.all(output.detach().numpy() >= 0.0)
    assert not np.isnan(output.detach().numpy()).any()
    assert not np.isinf(output.detach().numpy()).any()

    assert output.shape == input.shape[:1] + (out_channels, expected_size, expected_size)
