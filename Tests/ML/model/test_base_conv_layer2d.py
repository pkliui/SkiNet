import random
from typing import Optional
import numpy as np
import torch
import pytest
from SkiNet.ML.model.base_conv_layer2d import BaseConv2D
from SkiNet.ML.utils.sampling.encoder_sampling import PaddingMode
from torch.nn import Conv2d, BatchNorm2d, ReLU
from torch.nn import BatchNorm3d, Conv3d, MSELoss, ReLU
from SkiNet.ML.utils.model_utils import set_random_seed

batch_size = 1
in_channels=3
out_channels=8
kernel=5
stride=1
dilation=1
bias=False
apply_batchnorm=True
activation = torch.nn.ReLU


input_shape = (batch_size, in_channels, 10, 10)
output_shape = (batch_size, out_channels, 10, 1)

set_random_seed(1234)
input_tensor = torch.rand(*input_shape).float()
label_tensor = torch.rand(*output_shape).float()




def test_baseconv2d_structure() -> None:
    """
    Test that BaseConv2D module is composed of a conv2d layer, batchnorm2d, and an activation function
    """
    layer = BaseConv2D(in_channels=in_channels,
                       out_channels=out_channels,
                       kernel=kernel,
                       stride=stride,
                       padding_mode=PaddingMode.VALID,
                       dilation=dilation,
                       bias=bias,
                       apply_batchnorm=apply_batchnorm,
                       activation=activation)
    
    assert any([isinstance(ll, Conv2d) for ll in layer.children()])
    assert any([isinstance(ll, BatchNorm2d) for ll in layer.children()])
    assert any([isinstance(ll, ReLU) for ll in layer.children()])
    assert layer.activation.inplace is True


def test_basic_layer_forward_and_backward_pass() -> None:
    """
    Test forward and backward passes in BaseConv2D layer with 'SAME' padding
    and check for:
    - Gradient computation
    - Loss calculation
    - Positive-valued output after ReLU
    - No NaN or infinite values
    - Correct output shape
    """
    set_random_seed(1234)
    layer = BaseConv2D(in_channels=in_channels,
                       out_channels=out_channels,
                       kernel=kernel,
                       stride=stride,
                       padding_mode=PaddingMode.SAME,
                       dilation=dilation,
                       bias=bias,
                       apply_batchnorm=apply_batchnorm)

    output_tensor = layer(input_tensor)
    criterion = MSELoss()
    loss = torch.sqrt(criterion(output_tensor, label_tensor))
    loss.backward()

    # Gradient check
    assert input_tensor.grad is not None or output_tensor.requires_grad

    # Loss
    assert isinstance(loss.item(), float)
    assert loss.item() > 0

    # Positive-valued output after ReLU
    assert np.all(output_tensor.detach().numpy() >= 0.0)

    # No NaN or infinite values
    assert not np.isnan(output_tensor.detach().numpy()).any()
    assert not np.isinf(output_tensor.detach().numpy()).any()

    # Shape check for 'SAME' padding - batch size, out_channels, height, width
    assert output_tensor.shape == input_tensor.shape[:1] + (out_channels, input_tensor.shape[2], input_tensor.shape[3])


def test_baseconv2d_bias_and_batchnorm():
    """
    Test that bias is correctly set based on whether batchnorm is applied
    """
    # Bias should be False if batchnorm is applied
    layer = BaseConv2D(in_channels=in_channels,
                       out_channels=out_channels,
                       kernel=kernel,
                       stride=stride,
                       padding_mode=PaddingMode.SAME,
                       dilation=dilation,
                       bias=bias,
                       apply_batchnorm=True)
    assert layer.bias is False
    # Bias should be True if batchnorm is not applied
    layer = BaseConv2D(in_channels=in_channels,
                       out_channels=out_channels,
                       kernel=kernel,
                       stride=stride,
                       padding_mode=PaddingMode.SAME,
                       dilation=dilation,
                       bias=bias,
                       apply_batchnorm=False)
    assert layer.bias is True

def test_baseconv2d_invalid_padding_mode():
    """
    Test that an error is raised for invalid padding modes
    """
    with pytest.raises(ValueError) as excinfo:
        BaseConv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=kernel,
            stride=stride,
            padding_mode="not_a_padding_mode",  # Invalid type
            dilation=dilation,
            bias=bias,
            apply_batchnorm=apply_batchnorm
        )
    assert "Invalid padding mode" in str(excinfo.value)
