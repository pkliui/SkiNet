import numpy as np
import torch
import pytest
from torch.nn import Conv2d, BatchNorm2d, MSELoss

from SkiNet.ML.model.blocks.conv2d_layer import Conv2dLayer
from SkiNet.ML.utils.sampling.encoder_sampling import PaddingMode

@pytest.mark.parametrize("batch_size, in_channels, out_channels, kernel, padding_mode, dilation, apply_bias, apply_batchnorm, activation, input_size, expected_size", [
    (2, 3, 6, 5, PaddingMode.SAME, 1, False, True, torch.nn.ReLU, 10, 10),
    (2, 6, 12, 4, PaddingMode.DOWNSAMPLING_FACTOR_2, 1, False, True, torch.nn.ReLU, 10, 5),
])
def test_basic_layer_forward_and_backward_pass(batch_size, in_channels, out_channels, kernel, padding_mode, dilation, apply_bias, apply_batchnorm, activation, input_size, expected_size) -> None:
    """
    Test forward and backward passes in Conv2dLayer layer with 'SAME' padding
    """
    input = torch.rand(batch_size, in_channels, input_size, input_size).float()
    ground_truth = torch.rand(batch_size, out_channels, expected_size, expected_size).float()

    layer = Conv2dLayer(in_channels=in_channels,
                       out_channels=out_channels,
                       kernel=kernel,
                       padding_mode=padding_mode,
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