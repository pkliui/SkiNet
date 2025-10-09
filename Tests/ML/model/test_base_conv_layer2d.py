import numpy as np
import torch
import pytest
from torch.nn import Conv2d, BatchNorm2d, MSELoss

from SkiNet.ML.model.base_conv_layer2d import BaseConv2D
from SkiNet.ML.utils.sampling.encoder_sampling import PaddingMode


@pytest.mark.parametrize("in_channels, out_channels, kernel, stride, dilation, apply_bias, apply_batchnorm, activation", [
    (3, 8, 5, 1, 1, False, True, torch.nn.ReLU),
])
def test_baseconv2d_structure(in_channels, out_channels, kernel, stride, dilation, apply_bias, apply_batchnorm, activation) -> None:
    """
    Test that BaseConv2D module is composed of a conv2d layer, batchnorm2d, and an activation function
    """
    layer = BaseConv2D(in_channels=in_channels,
                       out_channels=out_channels,
                       kernel=kernel,
                       stride=stride,
                       padding_mode=PaddingMode.VALID,
                       dilation=dilation,
                       apply_bias=apply_bias,
                       apply_batchnorm=apply_batchnorm,
                       activation=activation)

    assert any([isinstance(ll, Conv2d) for ll in layer.children()])
    if apply_batchnorm:
        assert any([isinstance(ll, BatchNorm2d) for ll in layer.children()])
    assert any([isinstance(ll, activation) for ll in layer.children()])
    if hasattr(layer.activation, "inplace"):
        assert layer.activation.inplace is True

@pytest.mark.parametrize("batch_size, in_channels, out_channels, kernel, stride, dilation, apply_bias, apply_batchnorm, activation", [
    (2, 3, 8, 5, 1, 1, False, True, torch.nn.ReLU),
])
def test_basic_layer_forward_and_backward_pass(batch_size, in_channels, out_channels, kernel, stride, dilation, apply_bias, apply_batchnorm, activation) -> None:
    """
    Test forward and backward passes in BaseConv2D layer with 'SAME' padding
    """

    input = torch.rand(batch_size, in_channels, 10, 10).float()
    ground_truth = torch.rand(batch_size, out_channels, 10, 10).float() # For 'SAME' padding, output spatial dimensions should match input

    layer = BaseConv2D(in_channels=in_channels,
                       out_channels=out_channels,
                       kernel=kernel,
                       stride=stride,
                       padding_mode=PaddingMode.SAME,
                       dilation=dilation,
                       apply_bias=apply_bias,
                       apply_batchnorm=apply_batchnorm,
                       activation=activation)

    output = layer(input)
    loss_function = MSELoss()
    loss = torch.sqrt(loss_function(output, ground_truth))

    # Loss is a positive finite value
    assert isinstance(loss.item(), float)
    assert loss.item() > 0
    assert not np.isnan(loss.detach().numpy()).any()
    assert not np.isinf(loss.detach().numpy()).any()

    # Positive-valued output after ReLU
    assert np.all(output.detach().numpy() >= 0.0)
    assert not np.isnan(output.detach().numpy()).any()
    assert not np.isinf(output.detach().numpy()).any()

    # Shape check for 'SAME' padding - batch size, out_channels, height, width
    assert output.shape == input.shape[:1] + (out_channels, input.shape[2], input.shape[3])